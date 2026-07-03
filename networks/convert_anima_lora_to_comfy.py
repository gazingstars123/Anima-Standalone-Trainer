import argparse
from safetensors.torch import save_file
from safetensors import safe_open


from library import train_util
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

COMFYUI_DIT_PREFIX = "diffusion_model."
COMFYUI_QWEN3_PREFIX = "text_encoders.qwen3_06b.transformer.model."


def main(args):
    logger.info(f"Loading source file {args.src_path}")
    state_dict = {}
    with safe_open(args.src_path, framework="pt") as f:
        metadata = f.metadata()
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)

    logger.info(f"Converting...")

    keys = list(state_dict.keys())
    count = 0

    for k in keys:
        if not args.reverse:
            is_dit_lora = k.startswith("lora_unet_")
            if not is_dit_lora and not k.startswith("lora_te"):
                logger.warning(f"Skipping key with unrecognized prefix: {k}")
                state_dict.pop(k)
                continue
            module_and_weight_name = "_".join(k.split("_")[2:])  # Remove `lora_unet_`or `lora_te_` prefix

            if "." not in module_and_weight_name:
                logger.warning(f"Skipping malformed key (no separator dot): {k}")
                state_dict.pop(k)
                continue

            module_name, weight_name = module_and_weight_name.split(".", 1)

            if weight_name.startswith("lora_up"):
                weight_name = weight_name.replace("lora_up", "lora_B")
            elif weight_name.startswith("lora_down"):
                weight_name = weight_name.replace("lora_down", "lora_A")

            # _ → . then collapse multi-token attributes back via targeted replacements
            original_module_name = module_name.replace("_", ".")

            # DiT
            original_module_name = original_module_name.replace("llm.adapter", "llm_adapter")
            original_module_name = original_module_name.replace(".linear.", ".linear_")
            original_module_name = original_module_name.replace("t.embedding.norm", "t_embedding_norm")
            original_module_name = original_module_name.replace("x.embedder", "x_embedder")
            original_module_name = original_module_name.replace("t.embedder", "t_embedder")
            # Block attributes are single names (adaln_modulation_cross_attn, not nested dots)
            original_module_name = original_module_name.replace("adaln.modulation.cross.attn", "adaln_modulation_cross_attn")
            original_module_name = original_module_name.replace("adaln.modulation.self.attn", "adaln_modulation_self_attn")
            original_module_name = original_module_name.replace("adaln.modulation.mlp", "adaln_modulation_mlp")
            original_module_name = original_module_name.replace("cross.attn", "cross_attn")
            original_module_name = original_module_name.replace("k.proj", "k_proj")
            original_module_name = original_module_name.replace("k.norm", "k_norm")
            original_module_name = original_module_name.replace("q.proj", "q_proj")
            original_module_name = original_module_name.replace("q.norm", "q_norm")
            original_module_name = original_module_name.replace("v.proj", "v_proj")
            original_module_name = original_module_name.replace("o.proj", "o_proj")
            original_module_name = original_module_name.replace("output.proj", "output_proj")
            original_module_name = original_module_name.replace("self.attn", "self_attn")
            original_module_name = original_module_name.replace("final.layer", "final_layer")
            original_module_name = original_module_name.replace("adaln.modulation", "adaln_modulation")
            original_module_name = original_module_name.replace("out.proj", "out_proj")

            # Qwen3
            original_module_name = original_module_name.replace("embed.tokens", "embed_tokens")
            original_module_name = original_module_name.replace("input.layernorm", "input_layernorm")
            original_module_name = original_module_name.replace("down.proj", "down_proj")
            original_module_name = original_module_name.replace("gate.proj", "gate_proj")
            original_module_name = original_module_name.replace("up.proj", "up_proj")
            original_module_name = original_module_name.replace("post.attention.layernorm", "post_attention_layernorm")

            new_prefix = COMFYUI_DIT_PREFIX if is_dit_lora else COMFYUI_QWEN3_PREFIX
            new_k = f"{new_prefix}{original_module_name}.{weight_name}"
        else:
            if k.startswith(COMFYUI_DIT_PREFIX):
                is_dit_lora = True
                module_and_weight_name = k[len(COMFYUI_DIT_PREFIX) :]
            elif k.startswith(COMFYUI_QWEN3_PREFIX):
                is_dit_lora = False
                module_and_weight_name = k[len(COMFYUI_QWEN3_PREFIX) :]
            else:
                logger.warning(f"Skipping unrecognized key {k}")
                state_dict.pop(k)
                continue

            if "." not in module_and_weight_name:
                logger.warning(f"Skipping malformed key (no separator dot after prefix): {k}")
                state_dict.pop(k)
                continue

            # Dispatch on family-specific separator to avoid splitting the dotted module path
            if ".lora_" in module_and_weight_name:
                module_name, weight_name = module_and_weight_name.rsplit(".lora_", 1)
                weight_name = "lora_" + weight_name
            elif ".hada_" in module_and_weight_name:
                module_name, weight_name = module_and_weight_name.rsplit(".hada_", 1)
                weight_name = "hada_" + weight_name
            elif ".lokr_" in module_and_weight_name:
                module_name, weight_name = module_and_weight_name.rsplit(".lokr_", 1)
                weight_name = "lokr_" + weight_name
            else:
                module_name, weight_name = module_and_weight_name.rsplit(".", 1)  # alpha and other plain tensors

            if weight_name.startswith("lora_B"):
                weight_name = weight_name.replace("lora_B", "lora_up")
            elif weight_name.startswith("lora_A"):
                weight_name = weight_name.replace("lora_A", "lora_down")

            module_name = module_name.replace(".", "_")

            prefix = "lora_unet_" if is_dit_lora else "lora_te1_"

            new_k = f"{prefix}{module_name}.{weight_name}"

        state_dict[new_k] = state_dict.pop(k)
        count += 1

    logger.info(f"Converted {count} keys")
    if count == 0:
        logger.warning("No keys were converted. Please check if the source file is in the expected format.")
    elif count > 0 and count < len(keys):
        logger.warning(
            f"Only {count} out of {len(keys)} keys were converted. Please check if there are unexpected keys in the source file."
        )

    if metadata is not None:
        logger.info(f"Calculating hashes and creating metadata...")
        model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
        metadata["sshs_model_hash"] = model_hash
        metadata["sshs_legacy_hash"] = legacy_hash

    logger.info(f"Saving destination file {args.dst_path}")
    save_file(state_dict, args.dst_path, metadata=metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Anima LoRA/LoHa/LoKr format between sd-scripts and ComfyUI")
    parser.add_argument(
        "src_path",
        type=str,
        default=None,
        help="source path, sd-scripts format (or ComfyUI compatible format if --reverse is set, only supported for LoRAs converted by this script)",
    )
    parser.add_argument(
        "dst_path",
        type=str,
        default=None,
        help="destination path, ComfyUI compatible format (or sd-scripts format if --reverse is set)",
    )
    parser.add_argument("--reverse", action="store_true", help="reverse conversion direction")
    args = parser.parse_args()
    main(args)
