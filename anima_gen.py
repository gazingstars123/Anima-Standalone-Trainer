
import argparse
import os
import sys
import torch
import threading
from accelerate import Accelerator
from library import anima_utils, strategy_anima, anima_train_utils, train_util
from library.utils import setup_logging

setup_logging()
import logging
logger = logging.getLogger(__name__)

# Global models cache for server mode
CACHED_MODELS = None
APP_ACCELERATOR = None
CURRENT_LORA = {"path": None, "mul": 1.0}

def _assert_supported_network_module(lora_path):
    """anima_gen only wires up networks.lora_anima. A LoHa/LoKr checkpoint parses to 0
    adapters here and would merge as a no-op -> silent base-model output. Read the trained
    module from safetensors metadata and fail loud on a mismatch."""
    if not lora_path or not str(lora_path).endswith(".safetensors"):
        return
    try:
        from safetensors import safe_open
        with safe_open(lora_path, framework="pt") as f:
            mod = (f.metadata() or {}).get("ss_network_module")
    except Exception:
        return
    if mod and mod != "networks.lora_anima":
        raise ValueError(
            f"Checkpoint {lora_path} was trained with {mod}, but anima_gen supports "
            "only networks.lora_anima. LoHa/LoKr generation is not yet wired up; "
            "aborting instead of silently producing base-model output."
        )


def _assert_adapters_loaded(net, lora_path):
    """Universal catch for formats _assert_supported_network_module cannot see (no
    .safetensors metadata, e.g. .ckpt/.pt): a LoHa/LoKr checkpoint has no lora_down keys,
    so lora_anima builds an empty adapter set and merge becomes a no-op -> silent
    base-model output. Fail loud instead."""
    if not net.unet_loras and not net.text_encoder_loras:
        raise ValueError(
            f"0 adapters loaded from {lora_path}: checkpoint keys do not match "
            "networks.lora_anima (likely a LoHa/LoKr .ckpt/.pt). anima_gen would "
            "otherwise silently produce base-model output. Aborting."
        )


def _apply_lora(accelerator, models, lora_path, multiplier):
    _assert_supported_network_module(lora_path)
    import networks.lora_anima
    net, sd = networks.lora_anima.create_network_from_weights(
        multiplier=multiplier,
        file=lora_path,
        ae=models["vae"],
        text_encoders=[models["qwen3"]],
        unet=models["dit"],
        for_inference=True
    )
    _assert_adapters_loaded(net, lora_path)
    net.merge_to([models["qwen3"]], models["dit"], sd, models["dtype"], accelerator.device)
    del net

    # Sync secondary model for Parallel CFG.
    if models.get("dit_secondary") is not None:
        sec_device = next(models["dit_secondary"].parameters()).device
        net_sec, _ = networks.lora_anima.create_network_from_weights(
            multiplier=multiplier,
            file=lora_path,
            ae=models["vae"],
            text_encoders=[],
            unet=models["dit_secondary"],
            for_inference=True,
            weights_sd=sd,
        )
        net_sec.merge_to([], models["dit_secondary"], sd, models["dtype"], sec_device)
        del net_sec

    del sd
    torch.cuda.empty_cache()

def manage_lora(accelerator, models, target_path, target_mul):
    global CURRENT_LORA
    
    # Check if change is needed
    if CURRENT_LORA["path"] == target_path and abs(CURRENT_LORA["mul"] - target_mul) < 1e-6:
        return # No change
        
    # Unmerge current if exists
    if CURRENT_LORA["path"]:
        logger.info(f"Unmerging previous LoRA: {CURRENT_LORA['path']}")
        try:
            _apply_lora(accelerator, models, CURRENT_LORA["path"], -CURRENT_LORA["mul"])
            CURRENT_LORA["path"] = None
        except Exception as e:
            logger.error(f"Failed to unmerge LoRA: {e}")
            raise
            
    # Merge new if specified
    if target_path:
        logger.info(f"Merging new LoRA: {target_path} (x{target_mul})")
        try:
            _apply_lora(accelerator, models, target_path, target_mul)
            CURRENT_LORA["path"] = target_path
            CURRENT_LORA["mul"] = target_mul
        except Exception as e:
            logger.error(f"Failed to merge LoRA: {e}")
            raise

def load_models(args, accelerator):
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load Qwen3
    logger.info(f"Loading Qwen3 from {args.qwen3_path}")
    qwen3_text_encoder, _ = anima_utils.load_qwen3_text_encoder(
        args.qwen3_path, dtype=weight_dtype, device="cpu"
    )
    qwen3_text_encoder.eval()
    
    # Load DiT
    logger.info(f"Loading DiT from {args.dit_path}")
    transformer_dtype = None
    if hasattr(args, 'transformer_dtype') and args.transformer_dtype is not None:
         dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
         transformer_dtype = dtype_map.get(args.transformer_dtype, None)

    dit = anima_utils.load_anima_dit(
        args.dit_path,
        dtype=weight_dtype,
        device="cpu",
        transformer_dtype=transformer_dtype,
    )
    dit.eval()
    
    # Load VAE
    logger.info(f"Loading VAE from {args.vae_path}")
    vae, _, _, vae_scale = anima_utils.load_anima_vae(
        args.vae_path, dtype=torch.bfloat16, device="cpu"
    )
    vae.eval()

    # Load LoRA if specified (Merge into base models)
    if args.network_weights:
        # Initial LoRA load 
        logger.info(f"Loading LoRA weights from {args.network_weights}")
        _assert_supported_network_module(args.network_weights)
        import networks.lora_anima
        network, weights_sd = networks.lora_anima.create_network_from_weights(
            multiplier=args.network_mul,
            file=args.network_weights,
            ae=vae,
            text_encoders=[qwen3_text_encoder],
            unet=dit,
            for_inference=True
        )
        
        _assert_adapters_loaded(network, args.network_weights)
        network.merge_to([qwen3_text_encoder], dit, weights_sd, weight_dtype, "cpu")
        logger.info(f"LoRA merged with multiplier {args.network_mul}")
        
        global CURRENT_LORA
        CURRENT_LORA["path"] = args.network_weights
        CURRENT_LORA["mul"] = args.network_mul

    # Move to GPU (or distribute across GPUs)
    dit_secondary = None
    if args.device_map == 'parallel_cfg' and torch.cuda.device_count() > 1:
        # Parallel CFG: full model on each GPU for simultaneous pos/neg inference
        import copy
        logger.info("Parallel CFG mode: loading model on both GPUs")
        dit.to(torch.device('cuda:0'))
        logger.info("  Primary model on GPU 0")
        
        dit_secondary = copy.deepcopy(dit)
        dit_secondary.to(torch.device('cuda:1'))
        logger.info("  Secondary model (deepcopy) on GPU 1")
        
        qwen3_text_encoder.to(torch.device('cuda:0'))
        vae.to(torch.device('cuda:0'))
        vae_scale = [t.to(torch.device('cuda:0')) for t in vae_scale]
    elif args.device_map == 'sharding' and torch.cuda.device_count() > 1:
        # Multi-GPU model sharding: distribute DiT blocks across GPUs
        num_gpus = torch.cuda.device_count()
        logger.info(f"Multi-GPU sharding: distributing model across {num_gpus} GPUs")
        
        # Get VRAM for each GPU to split proportionally
        vram = []
        for i in range(num_gpus):
            total_mem = torch.cuda.get_device_properties(i).total_memory
            vram.append(total_mem)
            logger.info(f"  GPU {i}: {torch.cuda.get_device_properties(i).name} ({total_mem / 1e9:.1f} GB)")
        
        total_vram = sum(vram)
        num_blocks = len(dit.blocks)
        
        # Split blocks proportionally by VRAM
        gpu_assignments = []
        blocks_assigned = 0
        for i in range(num_gpus):
            if i == num_gpus - 1:
                n = num_blocks - blocks_assigned
            else:
                n = round(num_blocks * vram[i] / total_vram)
            gpu_assignments.append(n)
            blocks_assigned += n
        
        # Move components to their assigned GPUs
        first_gpu = torch.device('cuda:0')
        last_gpu = torch.device(f'cuda:{num_gpus - 1}')
        
        # Embeddings on first GPU, final layer on last GPU
        dit.x_embedder.to(first_gpu)
        dit.t_embedder.to(first_gpu)
        dit.t_embedding_norm.to(first_gpu)
        if hasattr(dit, 'pos_embedder'):
            dit.pos_embedder.to(first_gpu)
        if hasattr(dit, 'extra_pos_embedder'):
            dit.extra_pos_embedder.to(first_gpu)
        if hasattr(dit, 'llm_adapter'):
            dit.llm_adapter.to(first_gpu)
        dit.final_layer.to(last_gpu)
        
        # Distribute blocks across GPUs
        block_idx = 0
        for gpu_id in range(num_gpus):
            device = torch.device(f'cuda:{gpu_id}')
            n_blocks = gpu_assignments[gpu_id]
            for _ in range(n_blocks):
                dit.blocks[block_idx].to(device)
                block_idx += 1
            logger.info(f"  GPU {gpu_id}: {n_blocks} blocks")
        
        # Mark model as sharded so forward() knows to move tensors between devices
        dit._is_multi_gpu_sharded = True

        # Text encoder on first GPU, VAE on last GPU
        qwen3_text_encoder.to(first_gpu)
        vae.to(last_gpu)
        vae_scale = [t.to(last_gpu) for t in vae_scale]
    else:
        dit.to(accelerator.device)
        qwen3_text_encoder.to(accelerator.device)
        vae.to(accelerator.device)
        vae_scale = [t.to(accelerator.device) for t in vae_scale]

    # Enable specialized attention if requested
    if hasattr(args, 'flash_attn') and args.flash_attn:
        logger.info("Enabling Flash Attention for inference")
        dit.set_flash_attn(True)
        if dit_secondary is not None:
            dit_secondary.set_flash_attn(True)
    elif hasattr(args, 'sage_attn') and args.sage_attn:
        logger.info("Enabling SageAttention for inference")
        dit.set_sage_attn(True)
        if dit_secondary is not None:
            dit_secondary.set_sage_attn(True)

    return {
        "dit": dit,
        "dit_secondary": dit_secondary,
        "qwen3": qwen3_text_encoder,
        "vae": vae,
        "vae_scale": vae_scale,
        "tokenize_strategy": strategy_anima.AnimaTokenizeStrategy(
            qwen3_path=args.qwen3_path,
            qwen3_max_length=args.qwen3_max_token_length,
        ),
        "text_encoding_strategy": strategy_anima.AnimaTextEncodingStrategy(),
        "dtype": weight_dtype
    }

def perform_generation(args, models, accelerator):
    logger.info("Starting generation...")
    
    # Update args for specific request if needed (e.g. prompt file might have changed content)
    # The models are already loaded, so we just pass them
    
    anima_train_utils.sample_images(
        accelerator, args, 0, 0, 
        models["dit"], 
        models["vae"], 
        models["vae_scale"],
        models["qwen3"], 
        models["tokenize_strategy"], 
        models["text_encoding_strategy"],
        dit_secondary=models.get("dit_secondary"),
    )
    logger.info("Generation finished.")

def run_server(args, accelerator):
    global CACHED_MODELS
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        logger.error("Flask is required for server mode. Please install it.")
        sys.exit(1)

    app = Flask(__name__)
    
    # Load models once on startup
    CACHED_MODELS = load_models(args, accelerator)
    logger.info("Models loaded. Server ready.")

    import threading
    gen_lock = threading.Lock()

    @app.route('/generate', methods=['POST'])
    def handle_generate():
        if gen_lock.locked():
            return jsonify({"success": False, "error": "Generation already in progress. Please wait."}), 429

        with gen_lock:
            try:
                data = request.json
                if 'sample_prompts' in data:
                    args.sample_prompts = data['sample_prompts']
                
                if 'flow_shift' in data:
                    args.flow_shift = float(data['flow_shift'])
                
                # Handle Dynamic LoRA Switching
                req_weights = data.get('network_weights')
                req_mul = float(data.get('network_mul', 1.0))
                
                manage_lora(accelerator, CACHED_MODELS, req_weights, req_mul)

                # Handle Dynamic Attention Switching
                req_flash = bool(data.get('flash_attn', False))
                req_sage = bool(data.get('sage_attn', False))
                
                if 'dit' in CACHED_MODELS:
                    CACHED_MODELS['dit'].set_flash_attn(req_flash)
                    CACHED_MODELS['dit'].set_sage_attn(req_sage)
                if CACHED_MODELS.get('dit_secondary') is not None:
                    CACHED_MODELS['dit_secondary'].set_flash_attn(req_flash)
                    CACHED_MODELS['dit_secondary'].set_sage_attn(req_sage)

                perform_generation(args, CACHED_MODELS, accelerator)
                return jsonify({"success": True})
            except Exception as e:
                import traceback
                logger.error(f"Generation failed: {e}\n{traceback.format_exc()}")
                return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/ping', methods=['GET'])
    def ping():
        return jsonify({"status": "ready"})

    @app.route('/stop', methods=['POST'])
    def stop():
        func = request.environ.get('werkzeug.server.shutdown')
        if func:
            func()
        else:
            # Fallback for other servers or older werkzeug
            os._exit(0) 
        return jsonify({"success": True})

    app.run(host='0.0.0.0', port=args.server_port, debug=False, use_reloader=False)

def main():
    parser = argparse.ArgumentParser()
    # Add standard arguments
    train_util.add_dit_training_arguments(parser)
    anima_train_utils.add_anima_training_arguments(parser)
    
    # Add minimal others needed
    parser.add_argument("--sample_prompts", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--seed", type=int, default=None)
    
    # Fake args usually set by read_config_from_file or other places if missing
    parser.add_argument("--caption_dropout_rate", type=float, default=0.0)
    parser.add_argument("--sample_at_first", action="store_true", default=True)
    parser.add_argument("--sample_every_n_steps", type=int, default=None)
    parser.add_argument("--sample_every_n_epochs", type=int, default=None)
    parser.add_argument("--network_weights", type=str, default=None, help="Path to LoRA weights")
    parser.add_argument("--network_mul", type=float, default=1.0, help="LoRA multiplier")
    parser.add_argument("--device_map", type=str, default=None, help="Device map for model sharding (sharding = split across GPUs)")
    
    parser.add_argument("--sage_attn", action="store_true", help="Use SageAttention for inference (requires sageattention package)")
    parser.add_argument("--server_port", type=int, default=None, help="Run in server mode on this port")

    args = parser.parse_args()
    
    # Force sample_at_first to True for this script
    args.sample_at_first = True

    # Manual distributed initialization for Multi-GPU
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        import torch.distributed as dist
        if not dist.is_initialized():
            os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
            os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
            
            rank = int(os.environ.get("RANK", "0"))
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            
            if os.name == "nt":
                # Windows: use gloo and disable libuv for better stability
                os.environ["USE_LIBUV"] = "0"
                backend = "gloo"
            else:
                # Linux: use nccl for better performance
                backend = "nccl"
                
            logger.info(f"Initializing distributed process group (backend={backend}, rank={rank}, world_size={world_size})")
            dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    
    if args.server_port:
        logger.info(f"Starting Anima Generation Server on port {args.server_port}")
        run_server(args, accelerator)
    else:
        # Standard one-shot mode
        models = load_models(args, accelerator)
        perform_generation(args, models, accelerator)

if __name__ == "__main__":
    main()
