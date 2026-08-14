// === Anima Training UI — Client ===
const DEFAULT_NEGATIVE_PROMPT =
  "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia, low quality, worst quality, blurry, bad anatomy, extra limbs, deformed, watermark, text, signature, bareness, artifacts, hands, copyrights name, jpeg_artifacts, scan_artifacts, bad hands, missing fingers, extra digit, fewer digits, artistic error, ye-pop, deviantart, logo, patreon logo";
let currentJob = null;
let ws = null;
let isDirty = false;
let lastSavedConfig = null;
let lastSavedDataset = null;
let lastSavedPrompts = [];
let lastSavedNegativePrompt = "";
let samplesPollTimer = null;
let isDraggingBg = false;
let bgPosPercent = { x: 50, y: 50 };
let currentSubsets = [];
let archRegistry = null; // Loaded from /api/architectures
// --- DOM Refs ---
const $ = (id) => document.getElementById(id);
const jobListEl = $("job-list");
const emptyState = $("empty-state");
const jobEditor = $("job-editor");
const jobTitle = $("job-title");
const consoleOutput = $("console-output");
// ==========================================
//  API
// ==========================================
// Deletion API
async function deleteSamples(paths) {
  if (!currentJob) return;
  try {
    for (const fullPath of paths) {
      // Path is /api/jobs/:name/samples/samples/filename.png
      const parts = fullPath.split("/samples/");
      const relPath = parts[parts.length - 1];
      await fetch(`/api/jobs/${currentJob}/samples/${relPath}`, {
        method: "DELETE",
      });
    }
    // Remove from local state
    paths.forEach((p) => sampleState.selectedPaths.delete(p));
    // Refresh UI
    loadSamples();
  } catch (err) {
    console.error("Delete failed", err);
    showToast("Error deleting samples", "danger");
  }
}
async function api(url, opts = {}) {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const err = await res.json();
      msg = err.error || msg;
    } catch (_) { }
    throw new Error(msg);
  }
  return res.json();
}
// ==========================================
//  WebSocket
// ==========================================
function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}`);
  ws.onopen = () => {
    if (currentJob) {
      ws.send(JSON.stringify({ type: "subscribe", job: currentJob }));
    }
  };
  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      // hw_stats is global (not job-specific)
      if (msg.type === "hw_stats") {
        updateHwMonitor(msg.data);
        return;
      }
      if (msg.job !== currentJob) return;
      if (msg.type === "log") {
        appendConsole(msg.data);
      } else if (msg.type === "status") {
        if (msg.data === "generating") return; // Ignore generation status for Training button
        updateRunningState(msg.data === "running");
      }
    } catch (e) { }
  };
  ws.onclose = () => {
    setTimeout(connectWS, 3000);
  };
}
function subscribeToJob(jobName) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "subscribe", job: jobName }));
  }
}
// ==========================================
//  Hardware Monitor
// ==========================================
function formatHwBytes(bytes) {
  if (bytes >= 1073741824) return (bytes / 1073741824).toFixed(1) + " GB";
  if (bytes >= 1048576) return (bytes / 1048576).toFixed(0) + " MB";
  return bytes + " B";
}
function getTempClass(temp) {
  if (temp >= 80) return "hw-temp-hot";
  if (temp >= 65) return "hw-temp-warm";
  return "hw-temp-cool";
}
function updateHwMonitor(stats) {
  const container = document.getElementById("hw-stats-container");
  if (!container) return;
  const cpuPct = Math.max(0, Math.min(100, stats.cpu || 0));
  const ramPct = stats.ram
    ? Math.round((stats.ram.used / stats.ram.total) * 100)
    : 0;
  const ramUsed = stats.ram ? formatHwBytes(stats.ram.used) : "?";
  const ramTotal = stats.ram ? formatHwBytes(stats.ram.total) : "?";
  let html = "";
  const cpuTempHtml =
    stats.cpuTemp != null
      ? `<span class="hw-temp ${getTempClass(stats.cpuTemp)}">${stats.cpuTemp}°C</span>`
      : "";
  // System section (CPU + RAM)
  html += `<div class="hw-section">
        <div class="hw-section-header">
            <span class="hw-section-title">System</span>
            ${cpuTempHtml}
        </div>
        <div class="hw-row">
            <span class="hw-metric-label">CPU</span>
            <div class="hw-bar-wrap"><div class="hw-bar" style="width:${cpuPct}%"></div></div>
            <span class="hw-metric-value">${cpuPct}%</span>
        </div>
        <div class="hw-row">
            <span class="hw-metric-label">RAM</span>
            <div class="hw-bar-wrap"><div class="hw-bar hw-bar-ram" style="width:${ramPct}%"></div></div>
            <span class="hw-metric-value">${ramPct}% &nbsp;${ramUsed} / ${ramTotal}</span>
        </div>
    </div>`;
  // GPU sections
  if (stats.gpus && stats.gpus.length > 0) {
    stats.gpus.forEach((gpu) => {
      const gpuPct = Math.max(0, Math.min(100, gpu.util || 0));
      const vramPct =
        gpu.memTotal > 0 ? Math.round((gpu.memUsed / gpu.memTotal) * 100) : 0;
      const vramUsed = (gpu.memUsed / 1024).toFixed(1);
      const vramTotal = (gpu.memTotal / 1024).toFixed(1);
      const tempClass = getTempClass(gpu.temp);
      const activeClass = gpu.activity ? " hw-active" : "";
      const activityBadge = gpu.activity
        ? `<span class="hw-activity-badge">${gpu.activity}</span>`
        : "";
      const powerPct = gpu.powerLimit > 0 ? Math.round((gpu.powerDraw / gpu.powerLimit) * 100) : 0;
      const powerLabel = gpu.powerLimit > 0 ? `${gpu.powerDraw}W / ${gpu.powerLimit}W` : `${gpu.powerDraw}W`;
      html += `<div class="hw-section hw-section-gpu${activeClass}">
                <div class="hw-section-header">
                    <span class="hw-section-title">GPU ${gpu.index}</span>
                    <span class="hw-temp ${tempClass}">${gpu.temp}°C</span>
                    ${activityBadge}
                </div>
                <div class="hw-row">
                    <span class="hw-metric-label">Core</span>
                    <div class="hw-bar-wrap"><div class="hw-bar hw-bar-gpu" style="width:${gpuPct}%"></div></div>
                    <span class="hw-metric-value">${gpuPct}%</span>
                </div>
                <div class="hw-row">
                    <span class="hw-metric-label">VRAM</span>
                    <div class="hw-bar-wrap"><div class="hw-bar hw-bar-vram" style="width:${vramPct}%"></div></div>
                    <span class="hw-metric-value">${vramPct}% &nbsp;${vramUsed} / ${vramTotal} GB</span>
                </div>
                <div class="hw-row">
                    <span class="hw-metric-label">Power</span>
                    <div class="hw-bar-wrap"><div class="hw-bar hw-bar-power" style="width:${powerPct}%"></div></div>
                    <span class="hw-metric-value">${powerLabel}</span>
                </div>
            </div>`;
    });
  }
  container.innerHTML = html;
  // Compact bar (collapsed view)
  const compact = document.getElementById("hw-compact-bar");
  if (compact) {
    let ch = `<div class="hw-compact-item">
            <span class="hw-compact-label">CPU</span>
            <span>${cpuPct}%${stats.cpuTemp != null ? ` · <span class="hw-temp ${getTempClass(stats.cpuTemp)}">${stats.cpuTemp}°C</span>` : ""}</span>
        </div>
        <div class="hw-compact-item">
            <span class="hw-compact-label">RAM</span>
            <span>${ramUsed} / ${ramTotal}</span>
        </div>`;
    if (stats.gpus && stats.gpus.length > 0) {
      stats.gpus.forEach((gpu) => {
        const gpuPct = Math.max(0, Math.min(100, gpu.util || 0));
        const vramUsed = (gpu.memUsed / 1024).toFixed(1);
        const vramTotal = (gpu.memTotal / 1024).toFixed(1);
        const tempClass = getTempClass(gpu.temp);
        const badge = gpu.activity
          ? ` <span class="hw-activity-badge">${gpu.activity}</span>`
          : "";
        ch += `<div class="hw-compact-sep"></div>
                <div class="hw-compact-item">
                    <span class="hw-compact-label">GPU ${gpu.index}</span>
                    <span>${gpuPct}% · ${vramUsed}/${vramTotal}GB · ${gpu.powerDraw}W · <span class="hw-temp ${tempClass}">${gpu.temp}°C</span>${badge}</span>
                </div>`;
      });
    }
    compact.innerHTML = ch;
  }
}
// Keep tab-content padding in sync with monitor height + handle collapse
(function initHwMonitorResize() {
  const monitor = document.getElementById("hw-monitor");
  const toggleBtn = document.getElementById("hw-toggle");
  if (!monitor) return;
  function syncToggleArrow(isCollapsed) {
    if (toggleBtn)
      toggleBtn.textContent = isCollapsed ? "▲  Hardware Monitor" : "▼";
  }
  // Restore collapsed state
  const collapsed = localStorage.getItem("hw_monitor_collapsed") === "true";
  if (collapsed) monitor.classList.add("hw-collapsed");
  syncToggleArrow(collapsed);
  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      const isNowCollapsed = monitor.classList.toggle("hw-collapsed");
      localStorage.setItem("hw_monitor_collapsed", isNowCollapsed);
      syncToggleArrow(isNowCollapsed);
    });
  }
  if (!window.ResizeObserver) return;
  new ResizeObserver(() => {
    document.documentElement.style.setProperty(
      "--hw-bar-height",
      monitor.offsetHeight + 6 + "px",
    );
  }).observe(monitor);
})();
// ==========================================
//  Job List
// ==========================================
async function loadJobs() {
  const jobs = await api("/api/jobs");
  jobListEl.innerHTML = "";
  if (jobs.length === 0) {
    jobListEl.innerHTML =
      '<div style="padding:20px;text-align:center;color:var(--text-muted)">No jobs yet</div>';
    return;
  }
  jobs.forEach((job) => {
    const el = document.createElement("div");
    el.className = `job-item${job.name === currentJob ? " active" : ""}${job.running ? " running" : ""}`;
    el.innerHTML = `
            <div class="status-dot"></div>
            <span class="job-name">${job.name}</span>
        `;
    el.addEventListener("click", () => selectJob(job.name));
    jobListEl.appendChild(el);
  });
}
async function selectJob(name) {
  if (currentJob) savePromptTransientSettings();
  if (isDirty && !confirm("Unsaved changes. Switch anyway?")) return;
  isDirty = false;
  currentJob = name;
  if (samplesPollTimer) {
    clearInterval(samplesPollTimer);
    samplesPollTimer = null;
  }
  localStorage.setItem("lastJob", name);
  jobTitle.textContent = name;
  emptyState.classList.add("hidden");
  jobEditor.classList.remove("hidden");
  try {
    // Load available GPUs
    await loadGPUs();
    await loadGenGPUs();
    // Load job data
    const data = await api(`/api/jobs/${name}`);
    populateConfig(data.config);
    populateDataset(data.dataset);
    // Load prompts
    await loadPrompts();
    // Check run status
    const status = await api(`/api/jobs/${name}/train/status`);
    updateRunningState(status.running);
    // Set default negative prompt if no saved value exists for this job
    const savedTransient = localStorage.getItem(`prompt_transient_${name}`);
    if (!savedTransient || !JSON.parse(savedTransient).negative_prompt) {
      $("global-negative-prompt").value = DEFAULT_NEGATIVE_PROMPT;
    }
    // Save initial state for dirty checking
    lastSavedConfig = JSON.parse(JSON.stringify(gatherConfig()));
    lastSavedDataset = JSON.parse(JSON.stringify(gatherDataset()));
    lastSavedPrompts = JSON.parse(JSON.stringify(currentPrompts));
    lastSavedNegativePrompt = $("global-negative-prompt").value;
    // Subscribe WS
    subscribeToJob(name);
    // Reset console
    resetConsole();
    // Reset save button
    $("btn-save").classList.add("hidden");
    $("btn-discard").classList.add("hidden");
    // Refresh job list highlight
    loadJobs();
    // Load samples
    loadSamples();
    loadCheckpoints();
    loadPromptTransientSettings();
  } catch (err) {
    console.error(`Failed to load job "${name}":`, err);
    showToast(`Failed to load job: ${err.message}`, "danger");
    currentJob = null;
    localStorage.removeItem("lastJob");
    jobEditor.classList.add("hidden");
    emptyState.classList.remove("hidden");
  }
}
function updateRunningState(running) {
  $("btn-run").classList.toggle("hidden", running);
  $("btn-stop").classList.toggle("hidden", !running);
  // Update sidebar dot
  document.querySelectorAll(".job-item").forEach((el) => {
    const name = el.querySelector(".job-name").textContent;
    if (name === currentJob) {
      el.classList.toggle("running", running);
    }
  });
}
// ==========================================
//  Config UI Mapping
// ==========================================
function populateConfig(config) {
  const t = config.training_arguments || {};
  const n = config.network_arguments || {};
  const a = config.anima_arguments || {};
  // Training
  $("cfg-learning-rate").value = t.learning_rate || "5e-5";
  $("cfg-text-encoder-lr").value = t.text_encoder_lr || "5e-5";
  if (t.use_muon) {
    $("cfg-optimizer").value = "Muon";
  } else {
    $("cfg-optimizer").value = t.optimizer_type || "AdamW8bit";
  }
  $("cfg-lr-scheduler").value = t.lr_scheduler || "cosine";
  $("cfg-lr-warmup").value = t.lr_warmup_steps ?? 100;
  $("cfg-lr-scheduler-cycles").value = t.lr_scheduler_num_cycles ?? 1;
  $("cfg-lr-min-ratio").value = t.lr_scheduler_min_lr_ratio ?? 0;
  $("cfg-seed").value = t.seed ?? 42;
  // Extract weight decay
  let wdValue = "0";
  let decoupleValue = true;
  if (t.optimizer_args && Array.isArray(t.optimizer_args)) {
    const wdArg = t.optimizer_args.find((arg) =>
      String(arg).startsWith("weight_decay="),
    );
    if (wdArg) {
      wdValue = wdArg.split("=")[1];
    }
    const decoupleArg = t.optimizer_args.find((arg) =>
      String(arg).startsWith("decouple="),
    );
    if (decoupleArg) {
      decoupleValue = decoupleArg.split("=")[1].toLowerCase() === "true";
    }
  }
  $("cfg-weight-decay").value = wdValue;
  $("cfg-decouple").checked = decoupleValue;

  // Load Muon Settings
  $("cfg-muon-lr").value = t.muon_lr ?? 0.02;
  $("cfg-muon-lr-scale").value = t.muon_lr_scale ?? 0.05;
  $("cfg-muon-momentum").value = t.muon_momentum ?? 0.95;
  $("cfg-muon-weight-decay").value = t.muon_weight_decay ?? 0.01;
  $("cfg-muon-ns-steps").value = t.muon_ns_steps ?? 5;
  $("cfg-muon-param-filter").value = t.muon_param_filter || "self_attn_mlp_cross";
  $("cfg-muon-adam-lr").value = t.muon_adam_lr || "";
  $("cfg-muon-adam-betas").value = t.muon_adam_betas || "0.9,0.95";
  $("cfg-muon-adam-eps").value = t.muon_adam_eps ?? 1e-8;
  $("cfg-muon-disable-for-llm-adapter").checked = t.muon_disable_for_llm_adapter ?? true;
  $("cfg-muon-disable-distributed-allgather").checked = t.muon_disable_distributed_allgather ?? false;
  $("cfg-muon-fp32-sensitive").checked = t.muon_fp32_sensitive ?? false;

  // Update conditional visibility
  updateOptimizerOptions();
  updateLrSchedulerOptions();
  const maxSteps = t.max_train_steps;
  const isSteps = maxSteps && maxSteps > 0;
  document.querySelector(
    `input[name="duration-unit"][value="${isSteps ? "steps" : "epochs"}"]`,
  ).checked = true;
  updateDurationUnit();
  $("cfg-max-epochs").value = t.max_train_epochs ?? 20;
  $("cfg-save-every").value = t.save_every_n_epochs ?? 1;
  $("cfg-save-last-n-epochs").value = t.save_last_n_epochs ?? 0;
  $("cfg-keep-last-n-states-epochs").value = t.save_last_n_epochs_state ?? 1;
  $("cfg-max-steps").value = t.max_train_steps ?? 1000;
  $("cfg-save-every-steps").value = t.save_every_n_steps ?? 500;
  $("cfg-save-last-n-steps").value = t.save_last_n_steps ?? 0;
  $("cfg-keep-last-n-states-steps").value = t.save_last_n_steps_state ?? 1;
  $("cfg-output-name").value = t.output_name || "my_anima_lora";
  $("cfg-save-format").value = t.save_model_as || "safetensors";
  $("cfg-save-precision").value = t.save_precision || "bf16";
  $("cfg-mixed-precision").value = t.mixed_precision || "bf16";
  $("cfg-workers").value = t.max_data_loader_n_workers ?? 4;
  $("cfg-grad-acc").value = t.gradient_accumulation_steps ?? 1;
  $("cfg-gradient-checkpointing").checked = t.gradient_checkpointing ?? true;
  $("cfg-flash-attn").checked = t.flash_attn ?? false;
  $("cfg-torch-compile").checked = t.torch_compile ?? false;
  $("cfg-lowram").checked = t.lowram ?? false;
  $("cfg-blocks-to-swap").value = t.blocks_to_swap ?? 0;
  // Activation offload mode
  if (t.unsloth_offload_checkpointing) {
    $("cfg-activation-offload").value = "unsloth";
  } else if (t.cpu_offload_checkpointing) {
    $("cfg-activation-offload").value = "cpu";
  } else {
    $("cfg-activation-offload").value = "none";
  }
  updateActivationOffloadUI();
  $("cfg-persistent-workers").checked =
    t.persistent_data_loader_workers ?? true;
  $("cfg-cache-latents").checked = t.cache_latents_to_disk ?? true;
  $("cfg-vae-batch").value = t.vae_batch_size ?? 1;
  $("cfg-cache-te").checked = t.cache_text_encoder_outputs_to_disk ?? true;
  $("cfg-skip-cache-check").checked = t.skip_cache_check ?? false;
  $("cfg-disable-bucket-shuffle").checked = t.disable_bucket_shuffle ?? false;
  // Progressive resolution schedule
  if (t.resolution_schedule) {
    $("cfg-progressive-reso").checked = true;
    $("progressive-reso-panel").classList.remove("hidden");
    // Parse schedule string and populate fraction inputs after rendering phases
    window._pendingProgressiveSchedule = t.resolution_schedule;
  } else {
    $("cfg-progressive-reso").checked = false;
    $("progressive-reso-panel").classList.add("hidden");
    window._pendingProgressiveSchedule = null;
  }
  $("cfg-use-cuda-direct").checked = t.use_cuda_direct ?? false;
  $("cfg-ddp-gradient-as-bucket-view").checked =
    t.ddp_gradient_as_bucket_view ?? false;
  $("cfg-ddp-static-graph").checked = t.ddp_static_graph ?? false;
  $("cfg-ddp-find-unused-parameters").checked = t.ddp_find_unused_parameters ?? true;
  $("cfg-ddp-bucket-cap-mb").value = t.ddp_bucket_cap_mb ?? 25;
  // Multi-GPU mode selector (ddp/fsdp/fsdp2/deepspeed/tp_sp) — backward compat: infer from use_fsdp
  const restoredMode =
    t.multigpu_mode || (t.deepspeed ? "deepspeed" : t.use_fsdp ? "fsdp" : "ddp");
  $("cfg-multigpu-mode").value = restoredMode;
  applyMultiGpuMode(restoredMode);
  // TP/SP options
  if (t.tp_degree) $("cfg-tp-degree").value = t.tp_degree;
  if ($("cfg-tp-backend")) $("cfg-tp-backend").value = t.tp_backend || "auto";
  $("cfg-sequence-parallel").checked = true;
  if ($("cfg-no-fuse-qkv")) $("cfg-no-fuse-qkv").checked = t.no_fuse_qkv ?? false;
  $("cfg-fsdp-sharding-strategy").value = t.fsdp_sharding_strategy || "1";
  $("cfg-fsdp-offload-params").checked = t.fsdp_offload_params ?? false;
  $("cfg-fsdp-reshard-after-forward").checked =
    t.fsdp_reshard_after_forward ?? false;
  $("cfg-fsdp-activation-checkpointing").checked =
    t.fsdp_activation_checkpointing ?? false;
  $("cfg-fsdp-cpu-ram-efficient-loading").checked =
    t.fsdp_cpu_ram_efficient_loading ?? false;
  $("cfg-fsdp-backward-prefetch").value = t.fsdp_backward_prefetch || "";
  $("cfg-fsdp-forward-prefetch").checked = t.fsdp_forward_prefetch ?? false;
  $("cfg-fsdp-use-orig-params").checked = t.fsdp_use_orig_params ?? true;
  $("cfg-fsdp-limit-all-gathers").checked = t.fsdp_limit_all_gathers ?? true;
  $("cfg-fsdp-auto-wrap-policy").value = t.fsdp_auto_wrap_policy || "NO_WRAP";
  $("cfg-fsdp-min-num-params").value = t.fsdp_min_num_params || "100000000";
  $("cfg-fsdp-layer-to-wrap").value =
    t.fsdp_transformer_layer_cls_to_wrap || "";
  // fsdp-settings is always visible when group-fsdp is shown (mode dropdown controls visibility)
  $("fsdp-layer-wrap-group").classList.toggle(
    "hidden",
    $("cfg-fsdp-auto-wrap-policy").value !== "TRANSFORMER_BASED_WRAP",
  );
  $("fsdp-size-wrap-group").classList.toggle(
    "hidden",
    $("cfg-fsdp-auto-wrap-policy").value !== "SIZE_BASED_WRAP",
  );
  // FSDP2 restore
  $("cfg-fsdp2-reshard-after-forward").checked = t.fsdp2_reshard_after_forward ?? true;
  $("cfg-fsdp2-offload-params").checked = t.fsdp2_offload_params ?? false;
  $("cfg-fsdp2-activation-checkpointing").checked = t.fsdp2_activation_checkpointing ?? false;
  $("cfg-fsdp2-cpu-ram-efficient-loading").checked = t.fsdp2_cpu_ram_efficient_loading ?? false;
  $("cfg-fsdp2-auto-wrap-policy").value = t.fsdp2_auto_wrap_policy || "NO_WRAP";
  $("cfg-fsdp2-min-num-params").value = t.fsdp2_min_num_params || "100000000";
  $("cfg-fsdp2-layer-to-wrap").value = t.fsdp2_transformer_layer_cls_to_wrap || "";
  $("fsdp2-layer-wrap-group").classList.toggle(
    "hidden",
    $("cfg-fsdp2-auto-wrap-policy").value !== "TRANSFORMER_BASED_WRAP",
  );
  $("fsdp2-size-wrap-group").classList.toggle(
    "hidden",
    $("cfg-fsdp2-auto-wrap-policy").value !== "SIZE_BASED_WRAP",
  );
  // DeepSpeed restore
  $("cfg-ds-zero-stage").value = String(t.zero_stage ?? 2);
  $("cfg-ds-offload-optimizer-device").value =
    t.offload_optimizer_device || "none";
  $("cfg-ds-offload-optimizer-nvme-path").value =
    t.offload_optimizer_nvme_path || "";
  $("cfg-ds-offload-param-device").value = t.offload_param_device || "none";
  $("cfg-ds-offload-param-nvme-path").value =
    t.offload_param_nvme_path || "";
  $("cfg-ds-zero3-init-flag").checked = t.zero3_init_flag ?? false;
  $("cfg-ds-zero3-save-16bit-model").checked =
    t.zero3_save_16bit_model ?? false;
  $("cfg-ds-fp16-master-weights-and-gradients").checked =
    t.fp16_master_weights_and_gradients ?? false;
  updateDeepspeedOffloadUI();
  $("cfg-step-profile").checked = t.step_profile ?? false;
  $("cfg-profile-microbatch").checked = t.profile_microbatch ?? false;
  $("cfg-profile-microbatch-group").style.display = (t.step_profile ?? false) ? "" : "none";
  // Check GPU boxes based on config
  const savedIds = (config.gpu_ids || "")
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s);
  document.querySelectorAll('input[name="gpu-select"]').forEach((cb) => {
    cb.checked = savedIds.length === 0 || savedIds.includes(cb.value);
    // Also update card class if parent exists
    const card = cb.closest(".gpu-card");
    if (card) card.classList.toggle("selected", cb.checked);
  });
  updateMultiGPUUI();
  const sampleInterval = t.sample_every_n_epochs;
  const sampleIntervalSteps = t.sample_every_n_steps;
  const enableSampling =
    (sampleInterval !== null && sampleInterval > 0) ||
    (sampleIntervalSteps !== null && sampleIntervalSteps > 0);
  $("cfg-enable-sampling").checked = enableSampling;
  $("cfg-sample-every").value = sampleInterval || 1;
  $("cfg-sample-every-steps").value = sampleIntervalSteps || 100;
  $("group-sample-every").classList.toggle("hidden", !enableSampling);
  // Validation
  const enableValidation = (t.validation_split ?? 0) > 0;
  $("cfg-enable-validation").checked = enableValidation;
  $("group-validation").classList.toggle("hidden", !enableValidation);
  $("cfg-validation-split").value = t.validation_split || 0.1;
  $("cfg-validation-seed").value = t.validation_seed ?? 42;
  $("cfg-validate-every-epochs").value = t.validate_every_n_epochs ?? 1;
  $("cfg-validate-every-steps").value = t.validate_every_n_steps ?? 500;
  $("cfg-max-validation-steps").value = t.max_validation_steps ?? 10;
  // Anima
  $("cfg-timestep-method").value = a.timestep_sample_method || "logit_normal";
  $("cfg-flow-shift").value = a.discrete_flow_shift ?? 3.0;
  // Network / Training type
  const trainingType = n.network_module ? "lora" : "full_finetune";
  $("cfg-training-type").value = trainingType;
  updateTrainingTypeUI(trainingType);
  $("cfg-network-module").value = n.network_module || "networks.lora_anima";
  $("cfg-network-dim").value = n.network_dim ?? 16;
  $("cfg-network-alpha").value = n.network_alpha ?? 16;
  $("cfg-unet-only").checked = n.network_train_unet_only ?? true;
  $("cfg-network-weights").value = n.network_weights || "";
  $("cfg-freeze-llm-adapter").checked = t.freeze_llm_adapter ?? true;
  $("cfg-freeze-inserted-only-training").checked =
    t.freeze_inserted_only_training ?? false;
  $("cfg-auto-resume").checked = n.auto_resume_last_state ?? false;
  $("cfg-resume").value = n.resume || "";
  $("cfg-resume").disabled = $("cfg-auto-resume").checked;
  $("cfg-network-dropout").value = n.network_dropout ?? 0;
  $("cfg-network-args").value = (n.network_args || []).join(" ");
  // HuggingFace
  const hfEnabled = !!t.huggingface_repo_id;
  $("cfg-hf-enable").checked = hfEnabled;
  $("cfg-hf-repo-id").value = t.huggingface_repo_id || "";
  $("cfg-hf-path-in-repo").value = t.huggingface_path_in_repo || "";
  $("cfg-hf-visibility").value = t.huggingface_repo_visibility || "private";
  $("cfg-hf-save-state").checked = t.save_state_to_huggingface ?? false;
  $("cfg-hf-async").checked = t.async_upload ?? true;
  $("cfg-hf-token").value = t.huggingface_token || "";
  const hfResumeEnabled = !!t.resume_from_huggingface;
  $("cfg-hf-resume-enable").checked = hfResumeEnabled;
  $("cfg-hf-resume-path").value = (hfResumeEnabled && t.resume) ? t.resume : "";
  updateHfResumeUI(hfResumeEnabled);
  updateHfUI(hfEnabled);
}
function populateDataset(dataset) {
  const g = dataset.general || {};
  let dArray = [];
  if (Array.isArray(dataset.datasets)) {
    dArray = dataset.datasets;
  } else if (dataset.datasets) {
    dArray = [dataset.datasets];
  }
  if (dArray.length === 0) dArray = [{}];

  const resArray = dArray.map(d => Array.isArray(d.resolution) ? d.resolution[0] : (d.resolution || 1536));
  const batchArray = dArray.map(d => d.batch_size ?? 4);

  $("cfg-resolution").value = resArray.join(", ");
  $("cfg-batch-size").value = batchArray.join(", ");

  const d = dArray[0];
  $("cfg-caption-ext").value = d.caption_extension || ".txt";
  $("cfg-enable-bucket").checked = g.enable_bucket ?? true;
  $("cfg-bucket-no-upscale").checked = g.bucket_no_upscale ?? true;
  $("cfg-min-bucket").value = g.min_bucket_reso ?? 512;
  $("cfg-max-bucket").value = g.max_bucket_reso ?? 1536;
  $("cfg-bucket-steps").value = g.bucket_reso_steps ?? 64;
  // Load Subsets into memory
  let subsetsRaw = d.subsets || [];
  if (!Array.isArray(subsetsRaw)) subsetsRaw = [subsetsRaw];
  // Convert to our internal state format
  currentSubsets = subsetsRaw.map((s) => ({
    image_dir: s.image_dir || "",
    num_repeats: s.num_repeats ?? 1,
    keep_tokens: s.keep_tokens ?? 1,
    flip_aug: s.flip_aug ?? false,
    caption_prefix: s.caption_prefix || "",
    caption_dropout_rate: s.caption_dropout_rate ?? 0.05,
    caption_tag_dropout_rate: s.caption_tag_dropout_rate ?? 0.0,
    caption_dropout_every_n_epochs: s.caption_dropout_every_n_epochs ?? 0,
    shuffle_caption: s.shuffle_caption ?? false,
    cache_info: s.cache_info ?? true,
    is_reg: s.is_reg ?? false,
  }));
  // Edge case: if empty, force at least 1
  if (currentSubsets.length === 0) {
    addSubset(false);
  }
  // Alpha mask: check if any subset has it enabled
  $("cfg-alpha-mask").checked = subsetsRaw.some((s) => s.alpha_mask === true);
  renderSubsets();
  // Rebuild progressive phase rows now that resolution field is populated
  if ($("cfg-progressive-reso").checked) renderProgressivePhases();
}
function updateOptimizerOptions() {
  const optimizer = $("cfg-optimizer").value;
  const isProdigy =
    optimizer.includes("Prodigy") || optimizer.includes("DAdapt");
  $("group-decouple").classList.toggle("hidden", !isProdigy);

  // Toggle Muon Settings Group
  const isMuon = optimizer === "Muon";
  $("muon-settings-group").classList.toggle("hidden", !isMuon);

  if (isMuon) {
    // Muon is incompatible with DeepSpeed and FSDP1 (TP/SP is supported via
    // TPMuonWithAuxAdam)
    const mode = $("cfg-multigpu-mode")?.value;
    if (mode === "deepspeed" || mode === "fsdp") {
      showToast("Muon is incompatible with DeepSpeed and FSDP1. Reverted Multi-GPU mode to DDP.", "warning");
      if ($("cfg-multigpu-mode")) {
        $("cfg-multigpu-mode").value = "ddp";
        applyMultiGpuMode("ddp");
      }
    }
  }
}
function updateLrSchedulerOptions() {
  const scheduler = $("cfg-lr-scheduler").value;
  $("group-lr-scheduler-cycles").classList.toggle(
    "hidden",
    scheduler !== "cosine_with_restarts",
  );
  $("group-lr-min-ratio").classList.toggle(
    "hidden",
    scheduler !== "cosine_with_min_lr",
  );
}
function updateActivationOffloadUI() {
  const offload = $("cfg-activation-offload").value;
  const blocksInput = $("cfg-blocks-to-swap");
  const isOffload = offload !== "none";
  blocksInput.disabled = isOffload;
  if (isOffload) {
    blocksInput.value = 0;
  }
  // Auto-enable gradient checkpointing when offload is selected
  if (isOffload) {
    $("cfg-gradient-checkpointing").checked = true;
  }
}
// Helpers for safe parsing
function safeInt(val, fallback = 0) {
  if (val === "" || val === null || val === undefined) return fallback;
  const p = parseInt(val);
  return isNaN(p) ? fallback : p;
}
function safeFloat(val, fallback = 0.0) {
  if (val === "" || val === null || val === undefined) return fallback;
  const p = parseFloat(val);
  return isNaN(p) ? fallback : p;
}
function gatherConfig() {
  const unit = document.querySelector(
    'input[name="duration-unit"]:checked',
  ).value;
  const isEpochs = unit === "epochs";
  const enableSampling = $("cfg-enable-sampling").checked;
  const isMultiGpu =
    document.querySelectorAll('input[name="gpu-select"]:checked').length > 1;
  const multiGpuMode = $("cfg-multigpu-mode").value;
  const optimizerArgs = [];
  const wdValue = $("cfg-weight-decay").value;
  if (wdValue !== "") {
    optimizerArgs.push(`weight_decay=${wdValue}`);
  }
  if (!$("group-decouple").classList.contains("hidden")) {
    const isDecoupled = $("cfg-decouple").checked;
    optimizerArgs.push(`decouple=${isDecoupled ? "True" : "False"}`);
  }
  const config = {
    training_arguments: {
      output_name: $("cfg-output-name").value,
      save_model_as: $("cfg-save-format").value,
      max_train_epochs: isEpochs
        ? safeInt($("cfg-max-epochs").value)
        : undefined,
      save_every_n_epochs: isEpochs
        ? safeInt($("cfg-save-every").value)
        : undefined,
      save_last_n_epochs: isEpochs
        ? (safeInt($("cfg-save-last-n-epochs").value) || undefined)
        : undefined,
      save_last_n_epochs_state: isEpochs
        ? safeInt($("cfg-keep-last-n-states-epochs").value, 1)
        : undefined,
      sample_every_n_epochs:
        isEpochs && enableSampling
          ? safeInt($("cfg-sample-every").value)
          : undefined,
      max_train_steps: !isEpochs
        ? safeInt($("cfg-max-steps").value)
        : undefined,
      save_every_n_steps: !isEpochs
        ? safeInt($("cfg-save-every-steps").value)
        : undefined,
      save_last_n_steps: !isEpochs
        ? (safeInt($("cfg-save-last-n-steps").value) || undefined)
        : undefined,
      save_last_n_steps_state: !isEpochs
        ? safeInt($("cfg-keep-last-n-states-steps").value, 1)
        : undefined,
      sample_every_n_steps:
        !isEpochs && enableSampling
          ? safeInt($("cfg-sample-every-steps").value)
          : undefined,
      log_with: "tensorboard",
      learning_rate: safeFloat($("cfg-learning-rate").value),
      text_encoder_lr: safeFloat($("cfg-text-encoder-lr").value),
      optimizer_type: $("cfg-optimizer").value,
      optimizer_args: optimizerArgs.length > 0 ? optimizerArgs : undefined,
      // Muon configuration
      use_muon: $("cfg-optimizer").value === "Muon",
      muon_lr: safeFloat($("cfg-muon-lr").value),
      muon_lr_scale: safeFloat($("cfg-muon-lr-scale").value),
      muon_momentum: safeFloat($("cfg-muon-momentum").value),
      muon_weight_decay: safeFloat($("cfg-muon-weight-decay").value),
      muon_ns_steps: safeInt($("cfg-muon-ns-steps").value),
      muon_param_filter: $("cfg-muon-param-filter").value,
      muon_adam_lr: $("cfg-muon-adam-lr").value.trim() !== "" ? safeFloat($("cfg-muon-adam-lr").value) : undefined,
      muon_adam_betas: $("cfg-muon-adam-betas").value.trim(),
      muon_adam_eps: String($("cfg-muon-adam-eps").value).trim(),
      muon_disable_for_llm_adapter: $("cfg-muon-disable-for-llm-adapter").checked,
      muon_disable_for_adaln: true,
      muon_disable_distributed_allgather: $("cfg-muon-disable-distributed-allgather").checked,
      muon_fp32_sensitive: $("cfg-muon-fp32-sensitive").checked,
      lr_scheduler: $("cfg-lr-scheduler").value,
      lr_scheduler_num_cycles:
        $("cfg-lr-scheduler").value === "cosine_with_restarts"
          ? safeInt($("cfg-lr-scheduler-cycles").value)
          : undefined,
      lr_scheduler_min_lr_ratio:
        $("cfg-lr-scheduler").value === "cosine_with_min_lr"
          ? safeFloat($("cfg-lr-min-ratio").value)
          : undefined,
      lr_warmup_steps: safeInt($("cfg-lr-warmup").value),
      // Hardware
      mixed_precision: $("cfg-mixed-precision").value,
      save_precision: $("cfg-save-precision").value || undefined,
      max_data_loader_n_workers: safeInt($("cfg-workers").value),
      gradient_accumulation_steps: safeInt($("cfg-grad-acc").value),
      max_grad_norm: 1.0,
      gradient_checkpointing: $("cfg-gradient-checkpointing").checked,
      flash_attn: $("cfg-flash-attn").checked,
      torch_compile: $("cfg-torch-compile").checked,
      lowram: $("cfg-lowram").checked,
      blocks_to_swap: safeInt($("cfg-blocks-to-swap").value),
      ...($("cfg-activation-offload").value === "cpu" && {
        cpu_offload_checkpointing: true,
      }),
      ...($("cfg-activation-offload").value === "unsloth" && {
        unsloth_offload_checkpointing: true,
      }),
      persistent_data_loader_workers: $("cfg-persistent-workers").checked,
      seed: safeInt($("cfg-seed").value),
      ...($("cfg-enable-validation").checked && {
        validation_split: safeFloat($("cfg-validation-split").value),
      }),
      ...(isEpochs && $("cfg-enable-validation").checked
        ? { validate_every_n_epochs: safeInt($("cfg-validate-every-epochs").value) }
        : {}),
      ...(!isEpochs && $("cfg-enable-validation").checked
        ? { validate_every_n_steps: safeInt($("cfg-validate-every-steps").value) }
        : {}),
      ...($("cfg-enable-validation").checked && {
        max_validation_steps: safeInt($("cfg-max-validation-steps").value),
        validation_seed: safeInt($("cfg-validation-seed").value),
      }),
      cache_latents_to_disk: $("cfg-cache-latents").checked,
      vae_batch_size: safeInt($("cfg-vae-batch").value),
      cache_text_encoder_outputs_to_disk: $("cfg-cache-te").checked,
      skip_cache_check: $("cfg-skip-cache-check").checked,
      ...($("cfg-disable-bucket-shuffle").checked && {
        disable_bucket_shuffle: true,
      }),
      multigpu_mode: isMultiGpu ? multiGpuMode : "ddp",
      ...(multiGpuMode === "tp_sp" && isMultiGpu
          ? {
              tp_degree: safeInt($("cfg-tp-degree").value) || 2,
              tp_backend: $("cfg-tp-backend")?.value || "auto",
              sequence_parallel: true,
              ...($("cfg-no-fuse-qkv")?.checked ? { no_fuse_qkv: true } : {}),
            }
          : {}),
      ...(multiGpuMode === "deepspeed" && isMultiGpu
        ? {
            deepspeed: true,
            zero_stage: safeInt($("cfg-ds-zero-stage").value, 2),
            offload_optimizer_device:
              $("cfg-ds-offload-optimizer-device").value !== "none"
                ? $("cfg-ds-offload-optimizer-device").value
                : undefined,
            offload_optimizer_nvme_path:
              $("cfg-ds-offload-optimizer-device").value === "nvme"
                ? $("cfg-ds-offload-optimizer-nvme-path").value.trim() ||
                  undefined
                : undefined,
            offload_param_device:
              $("cfg-ds-offload-param-device").value !== "none"
                ? $("cfg-ds-offload-param-device").value
                : undefined,
            offload_param_nvme_path:
              $("cfg-ds-offload-param-device").value === "nvme"
                ? $("cfg-ds-offload-param-nvme-path").value.trim() || undefined
                : undefined,
            zero3_init_flag: $("cfg-ds-zero3-init-flag").checked,
            zero3_save_16bit_model: $("cfg-ds-zero3-save-16bit-model").checked,
            fp16_master_weights_and_gradients: $(
              "cfg-ds-fp16-master-weights-and-gradients",
            ).checked,
          }
        : { deepspeed: false }),
      use_cuda_direct: isMultiGpu ? $("cfg-use-cuda-direct").checked : false,
      ddp_gradient_as_bucket_view: isMultiGpu
        ? $("cfg-ddp-gradient-as-bucket-view").checked
        : false,
      ddp_static_graph: isMultiGpu ? $("cfg-ddp-static-graph").checked : false,
      ddp_find_unused_parameters: isMultiGpu ? $("cfg-ddp-find-unused-parameters").checked : false,
      ddp_bucket_cap_mb: isMultiGpu ? (safeInt($("cfg-ddp-bucket-cap-mb").value) || 25) : 25,
      // FSDP Configs
      use_fsdp: isMultiGpu
        ? multiGpuMode === "fsdp" || multiGpuMode === "fsdp2"
        : false,
      fsdp_sharding_strategy: $("cfg-fsdp-sharding-strategy").value,
      fsdp_offload_params: $("cfg-fsdp-offload-params").checked,
      fsdp_reshard_after_forward: $("cfg-fsdp-reshard-after-forward").checked,
      fsdp_activation_checkpointing: $("cfg-fsdp-activation-checkpointing")
        .checked,
      fsdp_cpu_ram_efficient_loading: $("cfg-fsdp-cpu-ram-efficient-loading")
        .checked,
      fsdp_backward_prefetch: $("cfg-fsdp-backward-prefetch").value,
      fsdp_forward_prefetch: $("cfg-fsdp-forward-prefetch").checked,
      fsdp_use_orig_params: $("cfg-fsdp-use-orig-params").checked,
      fsdp_limit_all_gathers: $("cfg-fsdp-limit-all-gathers").checked,
      fsdp_auto_wrap_policy: $("cfg-fsdp-auto-wrap-policy").value,
      fsdp_min_num_params: safeInt($("cfg-fsdp-min-num-params").value),
      fsdp_transformer_layer_cls_to_wrap: $(
        "cfg-fsdp-layer-to-wrap",
      ).value.trim(),
      // FSDP2 Configs
      fsdp2_reshard_after_forward: $("cfg-fsdp2-reshard-after-forward").checked,
      fsdp2_offload_params: $("cfg-fsdp2-offload-params").checked,
      fsdp2_activation_checkpointing: $("cfg-fsdp2-activation-checkpointing").checked,
      fsdp2_cpu_ram_efficient_loading: $("cfg-fsdp2-cpu-ram-efficient-loading").checked,
      fsdp2_auto_wrap_policy: $("cfg-fsdp2-auto-wrap-policy").value,
      fsdp2_min_num_params: safeInt($("cfg-fsdp2-min-num-params").value),
      fsdp2_transformer_layer_cls_to_wrap: $("cfg-fsdp2-layer-to-wrap").value.trim(),
      // FFT options
      freeze_llm_adapter: $("cfg-freeze-llm-adapter").checked,
      freeze_inserted_only_training: $("cfg-freeze-inserted-only-training").checked,
      // HuggingFace upload
      ...($("cfg-hf-enable").checked && $("cfg-hf-repo-id").value.trim() ? {
        huggingface_repo_id: $("cfg-hf-repo-id").value.trim(),
        huggingface_repo_type: "model",
        huggingface_repo_visibility: $("cfg-hf-visibility").value,
        ...($("cfg-hf-path-in-repo").value.trim() && { huggingface_path_in_repo: $("cfg-hf-path-in-repo").value.trim() }),
        save_state_to_huggingface: $("cfg-hf-save-state").checked,
        async_upload: $("cfg-hf-async").checked,
        ...($("cfg-hf-token").value.trim() && { huggingface_token: $("cfg-hf-token").value.trim() }),
      } : {}),
      // HuggingFace resume
      ...($("cfg-hf-resume-enable").checked && $("cfg-hf-resume-path").value.trim() ? {
        resume_from_huggingface: true,
        resume: $("cfg-hf-resume-path").value.trim(),
        ...($("cfg-hf-token").value.trim() && { huggingface_token: $("cfg-hf-token").value.trim() }),
      } : {}),
      // Diagnostics
      step_profile: $("cfg-step-profile").checked,
      profile_microbatch: $("cfg-profile-microbatch").checked,
      // Progressive resolution schedule
      ...(() => {
        if (!$("cfg-progressive-reso").checked) return {};
        const resList = ($("cfg-resolution").value || "1536").split(",").map(r => parseInt(r.trim())).filter(Boolean);
        const inputs = document.querySelectorAll(".prog-reso-frac");
        if (inputs.length === 0 || resList.length < 2) return {};
        const parts = resList.map((r, i) => {
          const frac = parseFloat(inputs[i]?.value || 0);
          return `${r}:${frac.toFixed(2)}`;
        });
        return { resolution_schedule: parts.join(",") };
      })(),
    },
    network_arguments: $("cfg-training-type").value === "full_finetune"
      ? {
          auto_resume_last_state: $("cfg-auto-resume").checked,
          ...($("cfg-resume").value && !$("cfg-auto-resume").checked && { resume: $("cfg-resume").value }),
        }
      : {
          network_module: $("cfg-network-module").value,
          network_dim: safeInt($("cfg-network-dim").value),
          network_alpha: safeInt($("cfg-network-alpha").value),
          network_train_unet_only: $("cfg-unet-only").checked,
          ...(safeFloat($("cfg-network-dropout").value) > 0 && {
            network_dropout: safeFloat($("cfg-network-dropout").value),
          }),
          ...($("cfg-network-args").value.trim() && {
            network_args: $("cfg-network-args").value.trim().split(/\s+/),
          }),
          ...($("cfg-network-weights").value && {
            network_weights: $("cfg-network-weights").value,
          }),
          auto_resume_last_state: $("cfg-auto-resume").checked,
          ...($("cfg-resume").value && !$("cfg-auto-resume").checked && { resume: $("cfg-resume").value }),
        },
    anima_arguments: {
      timestep_sample_method: $("cfg-timestep-method").value,
      discrete_flow_shift: safeFloat($("cfg-flow-shift").value),
      weighting_scheme: "logit_normal",
    },
    gpu_ids: Array.from(
      document.querySelectorAll('input[name="gpu-select"]:checked'),
    )
      .map((cb) => cb.value)
      .join(","),
  };
  return config;
}
function gatherDataset() {
  const res = safeInt($("cfg-resolution").value);
  return {
    general: {
      enable_bucket: $("cfg-enable-bucket").checked,
      bucket_no_upscale: $("cfg-bucket-no-upscale").checked,
      min_bucket_reso: safeInt($("cfg-min-bucket").value),
      max_bucket_reso: safeInt($("cfg-max-bucket").value),
      bucket_reso_steps: safeInt($("cfg-bucket-steps").value),
    },
    datasets: (() => {
      const resStr = $("cfg-resolution").value || "1536";
      const batchStr = $("cfg-batch-size").value || "4";

      const resList = resStr.split(",").map(r => safeInt(r.trim()));
      const batchListRaw = batchStr.split(",").map(b => safeInt(b.trim()));

      return resList.map((r, i) => {
        const b = batchListRaw[i] !== undefined ? batchListRaw[i] : batchListRaw[batchListRaw.length - 1];
        return {
          resolution: [r, r],
          batch_size: b,
          caption_extension: $("cfg-caption-ext").value,
          subsets: currentSubsets.map((s) => {
            const subset = {
              image_dir: s.image_dir,
              num_repeats: safeInt(s.num_repeats),
              keep_tokens: safeInt(s.keep_tokens),
              flip_aug: s.flip_aug,
              caption_prefix: s.caption_prefix,
              caption_dropout_rate: safeFloat(s.caption_dropout_rate),
              caption_tag_dropout_rate: safeFloat(s.caption_tag_dropout_rate),
              caption_dropout_every_n_epochs: safeInt(
                s.caption_dropout_every_n_epochs,
              ),
              shuffle_caption: s.shuffle_caption,
              cache_info: s.cache_info,
            };
            if (s.is_reg) subset.is_reg = true;
            if ($("cfg-alpha-mask").checked) subset.alpha_mask = true;
            return subset;
          }),
        };
      });
    })(),
  };
}
// ==========================================
//  Dataset Subsets
// ==========================================
function addSubset(shouldRender = true) {
  currentSubsets.push({
    image_dir: "",
    num_repeats: 1,
    keep_tokens: 1,
    flip_aug: false,
    caption_prefix: "",
    caption_dropout_rate: 0.05,
    caption_tag_dropout_rate: 0.0,
    caption_dropout_every_n_epochs: 0,
    shuffle_caption: false,
    cache_info: true,
    is_reg: false,
    collapsed: false,
  });
  if (shouldRender) {
    renderSubsets();
    checkDirty();
  }
}
function deleteSubset(idx) {
  if (currentSubsets.length <= 1) return; // Prevent deleting the last one
  currentSubsets.splice(idx, 1);
  renderSubsets();
  checkDirty();
}
function renderSubsets() {
  const container = $("dataset-subsets-list");
  if (!container) return;
  container.innerHTML = "";
  currentSubsets.forEach((subset, idx) => {
    const isLastOne = currentSubsets.length === 1;
    const isCollapsed = !!subset.collapsed;
    const card = document.createElement("div");
    card.className = "prompt-card-edit";
    card.style.flexDirection = "column";
    card.style.alignItems = "stretch";
    card.style.padding = isCollapsed ? "8px 15px" : "15px";
    const dirName = subset.image_dir
      ? subset.image_dir.split(/[\\/]/).pop()
      : "Empty Path";
    card.innerHTML = `
            <div class="prompt-card-header" style="justify-content: space-between; align-items: center; border-bottom: ${isCollapsed ? "none" : "1px solid var(--border)"}; padding-bottom: ${isCollapsed ? "0" : "8px"}; margin-bottom: ${isCollapsed ? "0" : "12px"};">
                <div style="display: flex; align-items: center; gap: 10px; cursor: pointer; flex: 1;" class="subset-toggle">
                    <span style="font-size: 0.8rem; transition: transform 0.2s; transform: rotate(${isCollapsed ? "-90deg" : "0deg"})">▼</span>
                    <label style="font-weight: 600; cursor: pointer;">Dataset ${idx + 1} ${subset.is_reg ? '<span style="font-size: 0.7rem; color: var(--text-muted); background: var(--border); padding: 1px 6px; border-radius: 4px; margin-left: 6px;">REG</span>' : ""}<span style="font-weight: normal; font-size: 0.8rem; color: var(--text-muted); margin-left: 10px;">${isCollapsed ? "(" + dirName + ")" : ""}</span></label>
                </div>
                <button class="btn btn-ghost btn-sm btn-delete-subset" title="Delete Dataset" 
                    ${isLastOne ? "disabled" : ""} 
                    style="color: var(--danger, #ff4d4d); transition: transform 0.1s; ${isLastOne ? "opacity:0.3; cursor:not-allowed;" : ""}">🗑️</button>
            </div>
            <div class="subset-body" style="display: ${isCollapsed ? "none" : "block"}">
                <div class="form-group">
                    <label style="font-size: 0.8rem;">Image Directory</label>
                    <div style="display: flex; gap: 8px;">
                        <input type="text" class="sub-image-dir" value="${escapeHtml(subset.image_dir)}" placeholder="C:\\path\\to\\images" style="flex: 1;">
                        <button class="btn btn-secondary btn-open-dir" title="Open folder">📂</button>
                    </div>
                </div>
                <div class="form-row" style="margin-top: 10px;">
                    <div class="form-group">
                        <label style="font-size: 0.8rem;">Num Repeats</label>
                        <input type="number" class="sub-num-repeats" value="${subset.num_repeats}" min="1">
                    </div>
                    <div class="form-group">
                        <label style="font-size: 0.8rem;">Keep Tokens</label>
                        <input type="number" class="sub-keep-tokens" value="${subset.keep_tokens}" min="0">
                    </div>
                </div>
                <div class="form-group" style="margin-top: 10px;">
                    <label style="font-size: 0.8rem;">Caption Prefix</label>
                    <input type="text" class="sub-caption-prefix" value="${escapeHtml(subset.caption_prefix)}" placeholder="e.g. A photo of,">
                </div>
                <div class="form-row" style="margin-top: 10px;">
                    <div class="form-group">
                        <label style="font-size: 0.8rem;">Caption Dropout Rate</label>
                        <input type="number" class="sub-caption-dropout" value="${subset.caption_dropout_rate}" step="0.01" min="0" max="1">
                    </div>
                    <div class="form-group">
                        <label style="font-size: 0.8rem;">Tag Dropout Rate</label>
                        <input type="number" class="sub-tag-dropout" value="${subset.caption_tag_dropout_rate}" step="0.01" min="0" max="1">
                    </div>
                </div>
                <div class="form-row" style="margin-top: 10px;">
                    <div class="form-group">
                        <label style="font-size: 0.8rem;">Dropout Every N Epochs</label>
                        <input type="number" class="sub-dropout-every-n" value="${subset.caption_dropout_every_n_epochs}" min="0">
                        <small style="display:block; font-size: 0.7rem; color: var(--text-muted);">0 = disabled</small>
                    </div>
                    <div class="form-group">
                    </div>
                </div>
                <div class="form-row" style="margin-top: 10px;">
                    <div class="form-group">
                        <label style="font-size: 0.8rem;"><input type="checkbox" class="sub-shuffle-caption" ${subset.shuffle_caption ? "checked" : ""}> Shuffle Captions</label>
                    </div>
                    <div class="form-group">
                        <label style="font-size: 0.8rem;"><input type="checkbox" class="sub-flip-aug" ${subset.flip_aug ? "checked" : ""}> Flip Augmentations</label>
                    </div>
                    <div class="form-group">
                        <label style="font-size: 0.8rem;"><input type="checkbox" class="sub-cache-info" ${subset.cache_info ? "checked" : ""}> Cache Metadata</label>
                    </div>
                </div>
                <div class="form-group" style="margin-top: 10px;">
                    <label style="font-size: 0.8rem;"><input type="checkbox" class="sub-is-reg" ${subset.is_reg ? "checked" : ""}> Regularization Dataset</label>
                    <small style="display:block; font-size: 0.7rem; color: var(--text-muted);">Images in this folder are used as regularization (class images) to prevent overfitting.</small>
                </div>
            </div>
        `;
    // Toggle collapse
    card.querySelector(".subset-toggle").addEventListener("click", () => {
      subset.collapsed = !subset.collapsed;
      renderSubsets();
    });
    // Update memory immediately on input
    if (!isCollapsed) {
      const updateSubset = () => {
        subset.image_dir = card.querySelector(".sub-image-dir").value;
        subset.num_repeats = safeInt(
          card.querySelector(".sub-num-repeats").value,
        );
        subset.keep_tokens = safeInt(
          card.querySelector(".sub-keep-tokens").value,
        );
        subset.caption_prefix = card.querySelector(".sub-caption-prefix").value;
        subset.caption_dropout_rate = safeFloat(
          card.querySelector(".sub-caption-dropout").value,
        );
        subset.caption_tag_dropout_rate = safeFloat(
          card.querySelector(".sub-tag-dropout").value,
        );
        subset.caption_dropout_every_n_epochs = safeInt(
          card.querySelector(".sub-dropout-every-n").value,
        );
        subset.shuffle_caption = card.querySelector(
          ".sub-shuffle-caption",
        ).checked;
        subset.flip_aug = card.querySelector(".sub-flip-aug").checked;
        subset.cache_info = card.querySelector(".sub-cache-info").checked;
        subset.is_reg = card.querySelector(".sub-is-reg").checked;
        checkDirty();
      };
      card.querySelectorAll("input").forEach((input) => {
        input.addEventListener("input", updateSubset);
        if (input.type === "checkbox") {
          input.addEventListener("change", updateSubset);
        }
      });
      card
        .querySelector(".btn-open-dir")
        .addEventListener("click", async () => {
          const dir = subset.image_dir.trim();
          if (!dir) {
            showToast("Please enter a directory path first");
            return;
          }
          const result = await api("/api/system/open-folder", {
            method: "POST",
            body: { path: dir },
          });
          if (result.error) {
            showToast("Error: " + result.error);
          }
        });
    }
    if (!isLastOne) {
      card
        .querySelector(".btn-delete-subset")
        .addEventListener("click", (e) => {
          e.stopPropagation(); // Don't trigger toggle
          deleteSubset(idx);
        });
    }
    container.appendChild(card);
  });
}
// ==========================================
//  Save
// ==========================================
async function saveJob() {
  if (!currentJob) return;
  const config = gatherConfig();
  const dataset = gatherDataset();
  // Prevent duplicate directories
  const subPaths = dataset.datasets[0].subsets
    .map((s) => s.image_dir.trim().toLowerCase())
    .filter((p) => p !== "");
  const uniquePaths = new Set(subPaths);
  if (uniquePaths.size !== subPaths.length) {
    showToast(
      "Error: Duplicate Image Directories detected. Each subset must have a unique path.",
    );
    return;
  }
  // Save Config & Dataset
  await api(`/api/jobs/${currentJob}`, {
    method: "PUT",
    body: { config, dataset },
  });
  // Save Prompts
  await savePrompts();
  // Update last saved state
  lastSavedConfig = JSON.parse(JSON.stringify(config));
  lastSavedDataset = JSON.parse(JSON.stringify(dataset));
  lastSavedPrompts = JSON.parse(JSON.stringify(currentPrompts));
  lastSavedNegativePrompt = $("global-negative-prompt").value;
  checkDirty();
  showToast("Job saved");
}
function checkDirty() {
  if (!currentJob) return;
  const currentConfig = gatherConfig();
  const currentDataset = gatherDataset();
  // Deep compare
  const configChanged =
    JSON.stringify(currentConfig) !== JSON.stringify(lastSavedConfig);
  const datasetChanged =
    JSON.stringify(currentDataset) !== JSON.stringify(lastSavedDataset);
  const promptsChanged =
    JSON.stringify(currentPrompts) !== JSON.stringify(lastSavedPrompts);
  const negPromptChanged =
    ($("global-negative-prompt").value || "") !==
    (lastSavedNegativePrompt || "");
  isDirty =
    configChanged || datasetChanged || promptsChanged || negPromptChanged;
  if (isDirty) {
    $("btn-save").classList.remove("hidden");
    $("btn-discard").classList.remove("hidden");
  } else {
    $("btn-save").classList.add("hidden");
    $("btn-discard").classList.add("hidden");
  }
}
function discardChanges() {
  if (!currentJob || !isDirty) return;
  showConfirm(
    "Discard Changes",
    "Discard all unsaved changes and revert to last saved state?",
    () => {
      populateConfig(lastSavedConfig);
      populateDataset(lastSavedDataset);
      currentPrompts = JSON.parse(JSON.stringify(lastSavedPrompts));
      renderPrompts();
      isDirty = false;
      $("btn-save").classList.add("hidden");
      $("btn-discard").classList.add("hidden");
      showToast("Changes discarded");
    },
  );
}
// Show/hide microbatch option depending on step profile checkbox
$("cfg-step-profile").addEventListener("change", (e) => {
  $("cfg-profile-microbatch-group").style.display = e.target.checked ? "" : "none";
  if (!e.target.checked) $("cfg-profile-microbatch").checked = false;
});

// Show/hide LoRA-specific fields based on training type
function updateTrainingTypeUI(type) {
  const isLora = type === "lora";
  $("lora-config-section").classList.toggle("hidden", !isLora);
  $("fft-config-section").classList.toggle("hidden", isLora);

  // Enforce Muon and LoRA restriction
  const muonOption = Array.from($("cfg-optimizer").options).find(o => o.value === "Muon");
  if (muonOption) {
    if (isLora) {
      muonOption.disabled = true;
      if ($("cfg-optimizer").value === "Muon") {
        $("cfg-optimizer").value = "AdamW8bit"; // fallback
        updateOptimizerOptions();
        showToast("Muon optimizer is not supported for LoRA training. Reverted to AdamW8bit.", "warning");
      }
    } else {
      muonOption.disabled = false;
    }
  }
}
$("cfg-training-type").addEventListener("change", (e) => {
  updateTrainingTypeUI(e.target.value);
  checkDirty();
});

// Disable manual resume path when auto-resume is enabled
$("cfg-auto-resume").addEventListener("change", (e) => {
  $("cfg-resume").disabled = e.target.checked;
  if (e.target.checked) $("cfg-resume").value = "";
});

function updateHfUI(enabled) {
  const g = $("group-hf");
  if (enabled) {
    g.classList.remove("hidden");
    g.style.display = "flex";
  } else {
    g.classList.add("hidden");
    g.style.display = "none";
  }
}
$("cfg-hf-enable").addEventListener("change", (e) => updateHfUI(e.target.checked));

function updateHfResumeUI(enabled) {
  const g = $("group-hf-resume");
  if (enabled) {
    g.classList.remove("hidden");
    g.style.display = "flex";
  } else {
    g.classList.add("hidden");
    g.style.display = "none";
  }
}
$("cfg-hf-resume-enable").addEventListener("change", (e) => updateHfResumeUI(e.target.checked));

// Mark dirty on any input change
document.addEventListener("input", (e) => {
  if (e.target.closest(".tab-content") && e.target.closest(".tab-pane")) {
    checkDirty();
  }
});
// ==========================================
//  Prompts
// ==========================================
let currentPrompts = []; // Array of objects { text, w, h, s, l, d }
async function loadPrompts() {
  if (!currentJob) return;
  const data = await api(`/api/jobs/${currentJob}/prompts`);
  // Parse strings into objects
  currentPrompts = (data.prompts || []).map((line) => parsePromptLine(line));
  renderPrompts();
}
function parsePromptLine(line) {
  // Defaults
  const p = { text: "", w: 832, h: 1216, s: 20, l: 7.5, d: 1, skip: false };
  // Check if skipped
  if (line.trim().startsWith("#")) {
    p.skip = true;
    line = line.trim().substring(1).trim();
  }
  // Extract params
  const paramRegex = /\s+--([whdsl])\s+(\S+)/g;
  let match;
  while ((match = paramRegex.exec(line)) !== null) {
    const val = match[2];
    if (match[1] === "w") p.w = parseInt(val);
    if (match[1] === "h") p.h = parseInt(val);
    if (match[1] === "s") p.s = parseInt(val);
    if (match[1] === "d") p.d = parseInt(val);
    if (match[1] === "l") p.l = parseFloat(val);
  }
  // Extract text (strip out specific params and the negative prompt string)
  p.text = line
    .replace(/\s+--n\s+.*$/i, "") // Remove global negative prompt and everything after it
    .replace(/\s+--[whdsl]\s+\S+/gi, "") // Remove regular parameter flags
    .trim();
  return p;
}
function serializePrompt(p) {
  // Reconstruct line, ensuring no newlines break the backend parsing parser
  const safeText = p.text.replace(/[\r\n]+/g, " ").trim();
  let line = `${safeText} --w ${p.w} --h ${p.h} --s ${p.s} --d ${p.d} --l ${p.l}`;
  // Append global negative prompt without newlines
  const neg = $("global-negative-prompt")
    .value.replace(/[\r\n]+/g, " ")
    .trim();
  if (neg) {
    line += ` --n ${neg}`;
  }
  return p.skip ? `# ${line}` : line;
}
async function savePrompts() {
  // Filter out prompts that have no text before saving
  const validPrompts = currentPrompts.filter(
    (p) => p.text && p.text.trim().length > 0,
  );
  const lines = validPrompts.map(serializePrompt);
  await api(`/api/jobs/${currentJob}/prompts`, {
    method: "PUT",
    body: { prompts: lines },
  });
}
function renderPrompts() {
  const list = $("prompts-list");
  const empty = $("prompts-empty");
  if (currentPrompts.length === 0) {
    list.classList.add("hidden");
    empty.classList.remove("hidden");
    return;
  }
  empty.classList.add("hidden");
  list.classList.remove("hidden");
  list.innerHTML = "";
  currentPrompts.forEach((p, idx) => {
    const card = document.createElement("div");
    card.className = `prompt-card-edit${p.skip ? " skipped" : ""}`;
    card.innerHTML = `
            <div class="prompt-card-header">
                <label class="skip-label">
                    <input type="checkbox" class="p-skip" ${p.skip ? "checked" : ""}> Skip
                </label>
            </div>
            <textarea class="p-text" rows="2" placeholder="Enter prompt text...">${escapeHtml(p.text)}</textarea>
            <div class="prompt-card-row">
                <div class="compact-input">
                    <label>W</label>
                    <input type="number" class="p-w" value="${p.w}" step="64">
                </div>
                <div class="compact-input">
                    <label>H</label>
                    <input type="number" class="p-h" value="${p.h}" step="64">
                </div>
                <div class="compact-input">
                    <label>Steps</label>
                    <input type="number" class="p-s" value="${p.s}">
                </div>
                <div class="compact-input">
                    <label>Scale</label>
                    <input type="number" class="p-l" value="${p.l}" step="0.5">
                </div>
                <div class="compact-input">
                    <label>Seed</label>
                    <input type="number" class="p-d" value="${p.d}">
                </div>
                <button class="btn btn-ghost btn-sm btn-delete-prompt" title="Delete">🗑️</button>
            </div>
        `;
    // Bind events
    const updateState = () => {
      p.skip = card.querySelector(".p-skip").checked;
      p.text = card.querySelector(".p-text").value;
      p.w = parseInt(card.querySelector(".p-w").value);
      p.h = parseInt(card.querySelector(".p-h").value);
      p.s = parseInt(card.querySelector(".p-s").value);
      p.l = parseFloat(card.querySelector(".p-l").value);
      p.d = parseInt(card.querySelector(".p-d").value);
      card.classList.toggle("skipped", p.skip);
      checkDirty();
    };
    const tx = card.querySelector(".p-text");
    const autoResize = () => {
      tx.style.height = "auto";
      tx.style.height = tx.scrollHeight + 2 + "px";
    };
    tx.addEventListener("input", autoResize);
    // Initial resize
    setTimeout(autoResize, 1);
    card.querySelectorAll("input, textarea").forEach((el) => {
      el.addEventListener("input", updateState);
    });
    card
      .querySelector(".btn-delete-prompt")
      .addEventListener("click", () => deletePrompt(idx));
    list.appendChild(card);
  });
}
function deletePrompt(idx) {
  currentPrompts.splice(idx, 1);
  renderPrompts();
  checkDirty();
}
function addPrompt() {
  // Get defaults from global bar
  const w = parseInt($("global-w").value) || 832;
  const h = parseInt($("global-h").value) || 1216;
  const s = parseInt($("global-s").value) || 28;
  const l = parseFloat($("global-l").value) || 3.5;
  let d = parseInt($("global-d").value);
  // If global seed is 0 or empty, randomize for the new prompt
  if (!d || d === 0) {
    d = Math.floor(Math.random() * 99999) + 1;
  }
  currentPrompts.push({ text: "", w, h, s, l, d, skip: false });
  renderPrompts();
  checkDirty();
}
function applyGlobalSettings() {
  const w = parseInt($("global-w").value);
  const h = parseInt($("global-h").value);
  const s = parseInt($("global-s").value);
  const l = parseFloat($("global-l").value);
  const d = parseInt($("global-d").value);
  currentPrompts.forEach((p) => {
    if (w) p.w = w;
    if (h) p.h = h;
    if (s) p.s = s;
    if (l) p.l = l;
    // Seed handling: 0 = random for each prompt, non-zero = apply same seed to all
    if (d === 0) {
      p.d = Math.floor(Math.random() * 99999) + 1; // Random seed 1-99999
    } else if (d) {
      p.d = d;
    }
  });
  renderPrompts();
  renderPrompts();
  checkDirty();
  showToast(
    d === 0
      ? "Random seeds applied to all prompts"
      : "Global settings applied to all prompts",
  );
}
// === Prompt Tab Persistence ===
function savePromptTransientSettings() {
  if (!currentJob) return;
  const settings = {
    lora_mul: $("gen-lora-mul").value,
    keep_loaded: $("chk-keep-loaded").checked,
    flash_attn: $("gen-flash-attn").checked,
    sage_attn: $("gen-sage-attn").checked,
    global_w: $("global-w").value,
    global_h: $("global-h").value,
    global_s: $("global-s").value,
    global_l: $("global-l").value,
    global_d: $("global-d").value,
    selected_lora: $("gen-lora-select").value,
    negative_prompt: $("global-negative-prompt").value,
    gen_gpu_ids: getSelectedGenGPUs(),
    gen_multi_gpu_mode: $("gen-multi-gpu-mode").value,
  };
  localStorage.setItem(
    `prompt_transient_${currentJob}`,
    JSON.stringify(settings),
  );
}
function loadPromptTransientSettings() {
  if (!currentJob) return;
  const data = localStorage.getItem(`prompt_transient_${currentJob}`);
  if (!data) return;
  try {
    const settings = JSON.parse(data);
    if (settings.lora_mul !== undefined)
      $("gen-lora-mul").value = settings.lora_mul;
    if (settings.keep_loaded !== undefined)
      $("chk-keep-loaded").checked = settings.keep_loaded;
    if (settings.flash_attn !== undefined)
      $("gen-flash-attn").checked = settings.flash_attn;
    if (settings.sage_attn !== undefined)
      $("gen-sage-attn").checked = settings.sage_attn;
    if (settings.global_w !== undefined)
      $("global-w").value = settings.global_w;
    if (settings.global_h !== undefined)
      $("global-h").value = settings.global_h;
    if (settings.global_s !== undefined)
      $("global-s").value = settings.global_s;
    if (settings.global_l !== undefined)
      $("global-l").value = settings.global_l;
    if (settings.global_d !== undefined)
      $("global-d").value = settings.global_d;
    if (settings.negative_prompt !== undefined)
      $("global-negative-prompt").value = settings.negative_prompt;
    // Restore gen GPU selection
    if (settings.gen_gpu_ids !== undefined) {
      restoreGenGPUSelection(settings.gen_gpu_ids);
    }
    if (settings.gen_multi_gpu_mode !== undefined) {
      $("gen-multi-gpu-mode").value = settings.gen_multi_gpu_mode;
    }
    // selected_lora is handled in loadCheckpoints
  } catch (e) { }
}
// ==========================================
//  Console
// ==========================================
const CONSOLE_MAX_LINES = 3000;
let consoleQueue = [];
let consoleFlushScheduled = false;
let consoleTailRaw = ""; 
let consoleTailEl = null;
let consoleLineCount = 0;

function processCarriageReturns(segment) {
  if (!segment.includes("\r")) return segment;
  const pendingCR = segment.endsWith("\r");
  const lines = segment.split("\n");
  for (let i = 0; i < lines.length; i++) {
    if (!lines[i].includes("\r")) continue;
    const parts = lines[i].split("\r");
    let out = "";
    for (let j = 0; j < parts.length; j++) {
      if (j === parts.length - 1) {
        out += parts[j];
      } else if (parts[j + 1].length > 0) {
        out = ""; // Overwrite triggered by following content
      } else {
        out = parts[j]; // Keep until something actually follows the \r
      }
    }
    lines[i] = out;
  }
  return lines.join("\n") + (pendingCR ? "\r" : "");
}

function resetConsole() {
  consoleQueue = [];
  consoleFlushScheduled = false;
  consoleTailRaw = "";
  consoleTailEl = null;
  consoleLineCount = 0;
  if (consoleOutput) consoleOutput.textContent = "Waiting for training to start...";
}

function consoleCreateLineEl() {
  const div = document.createElement("div");
  consoleOutput.appendChild(div);
  consoleLineCount++;
  return div;
}

function consoleTrim() {
  while (consoleLineCount > CONSOLE_MAX_LINES && consoleOutput.firstElementChild) {
    consoleOutput.firstElementChild.remove();
    consoleLineCount--;
  }
}

function consoleFlush() {
  consoleFlushScheduled = false;
  if (!consoleOutput || !consoleQueue.length) return;
  const newText = consoleQueue.join("");
  consoleQueue = [];
  // First real content after reset — clear the placeholder text node.
  if (consoleOutput.childElementCount === 0) consoleOutput.textContent = "";

  const wasNearBottom =
    consoleOutput.scrollHeight -
    consoleOutput.scrollTop -
    consoleOutput.clientHeight <
    100;

  const processed = processCarriageReturns(consoleTailRaw + newText);
  const parts = processed.split("\n");

  for (let i = 0; i < parts.length - 1; i++) {
    if (!consoleTailEl) consoleTailEl = consoleCreateLineEl();
    consoleTailEl.textContent = parts[i];
    consoleTailEl = null;
  }
  consoleTailRaw = parts[parts.length - 1];

  const tailRenderText = consoleTailRaw.endsWith("\r") ? consoleTailRaw.slice(0, -1) : consoleTailRaw;
  if (!consoleTailEl) consoleTailEl = consoleCreateLineEl();
  consoleTailEl.textContent = tailRenderText;

  consoleTrim();
  if (wasNearBottom) {
    consoleOutput.scrollTop = consoleOutput.scrollHeight;
  }
}

function appendConsole(text) {
  if (!consoleOutput || !text) return;
  consoleQueue.push(text);
  if (!consoleFlushScheduled) {
    consoleFlushScheduled = true;
    requestAnimationFrame(consoleFlush);
  }
}
// ==========================================
//  Samples
// ==========================================
async function loadCheckpoints() {
  if (!currentJob) return;
  const jobAtStart = currentJob;
  const files = await api(`/api/jobs/${currentJob}/checkpoints`);
  if (currentJob !== jobAtStart) return; // job changed while fetching
  const select = $("gen-lora-select");
  // Save current selection
  const currentVal = select.value;
  select.innerHTML = '<option value="">Base Model (No LoRA)</option>';
  files.forEach((f) => {
    const opt = document.createElement("option");
    opt.value = f.path;
    opt.textContent = `${f.name} (${new Date(f.mtime).toLocaleString()})`;
    select.appendChild(opt);
  });
  // Restore selection if exists
  const data = localStorage.getItem(`prompt_transient_${currentJob}`);
  let savedLora = null;
  if (data) {
    try {
      savedLora = JSON.parse(data).selected_lora;
    } catch (e) { }
  }
  const valToRestore = currentVal || savedLora;
  if (
    valToRestore &&
    Array.from(select.options).some((o) => o.value === valToRestore)
  ) {
    select.value = valToRestore;
  }
}
// Sample State
let sampleState = {
  selectedPaths: new Set(),
  lastSelectedPath: null,
  groups: {}, // enum -> [images]
  allImages: [], // flat list for index lookup
  isExplicitMultiSelect: false,
  expandedGroups: new Set(), // group keys the user chose to "show all" for
  samplesJob: null, // job the groups/expansion state belongs to
};
const SAMPLES_LIMIT_OPTIONS = ["5", "10", "20", "50", "all"];
function getSamplesLimit() {
  const saved = localStorage.getItem("samples_limit");
  return SAMPLES_LIMIT_OPTIONS.includes(saved) ? saved : "10";
}
function setSamplesLimit(value) {
  localStorage.setItem("samples_limit", value);
}
async function loadSamples(isUpdate = false) {
  if (!currentJob) return;
  if (sampleState.samplesJob !== currentJob) {
    sampleState.samplesJob = currentJob;
    sampleState.expandedGroups.clear();
  }
  const images = await api(`/api/jobs/${currentJob}/samples`);
  const container = $("samples-grid");
  const empty = $("samples-empty");
  if (!images || images.length === 0) {
    if (!isUpdate) {
      container.classList.add("hidden");
      empty.classList.remove("hidden");
    }
    return;
  }
  empty.classList.add("hidden");
  container.classList.remove("hidden");
  renderSampleGroups(images);
}
function renderSampleGroups(images) {
  const container = $("samples-grid");
  // Load manual order from localStorage
  const savedOrder = loadManualOrder();
  const orderMap = new Map();
  if (savedOrder) {
    savedOrder.forEach((item, index) => {
      orderMap.set(item.path, { group: item.group, index: index });
    });
  }
  // 1. Group Images
  const groups = {};
  images.forEach((img) => {
    let groupKey;
    if (orderMap.has(img.path)) {
      groupKey = orderMap.get(img.path).group;
    } else {
      const match = img.name.match(/_(\d{2,})_\d{14}/);
      groupKey = match ? match[1] : "default";
    }
    if (!groups[groupKey]) groups[groupKey] = [];
    groups[groupKey].push(img);
  });
  sampleState.groups = groups;
  sampleState.allImages = images;
  const limitSetting = getSamplesLimit();
  const limit = limitSetting === "all" ? Infinity : parseInt(limitSetting, 10);
  // 2. Render Groups
  container.innerHTML = "";
  const sortedGroupKeys = Object.keys(groups).sort((a, b) => {
    if (a === "default") return 1;
    if (b === "default") return -1;
    return parseInt(a) - parseInt(b);
  });
  sortedGroupKeys.forEach((key) => {
    const groupDiv = document.createElement("div");
    groupDiv.className = "sample-group";
    groupDiv.dataset.group = key;
    const header = document.createElement("div");
    header.className = "group-header";
    header.textContent =
      key === "default" ? "Uncategorized" : `Prompt ${parseInt(key) + 1}`;
    const gridDiv = document.createElement("div");
    gridDiv.className = "group-grid";
    gridDiv.addEventListener("dragover", handleDragOver);
    gridDiv.addEventListener("drop", handleDrop);
    // Sort images in group.
    // If they have a saved index, use it. Otherwise, use mtime (newest first).
    groups[key].sort((a, b) => {
      const orderA = orderMap.get(a.path);
      const orderB = orderMap.get(b.path);
      if (orderA && orderB) return orderA.index - orderB.index;
      if (orderA) return 1; // Saved items come after new items?
      if (orderB) return -1;
      return b.mtime - a.mtime; // Default newest first for items without saved order
    });
    const total = groups[key].length;
    const expanded = sampleState.expandedGroups.has(key);
    const visible = expanded ? groups[key] : groups[key].slice(0, limit);
    visible.forEach((img) => {
      createSampleCard(img, gridDiv);
    });
    groupDiv.appendChild(header);
    groupDiv.appendChild(gridDiv);
    const hiddenCount = total - visible.length;
    if (hiddenCount > 0) {
      const showMore = document.createElement("div");
      showMore.className = "group-show-more";
      showMore.textContent = `Show all (${hiddenCount} more)`;
      showMore.addEventListener("click", () => {
        sampleState.expandedGroups.add(key);
        renderSampleGroups(sampleState.allImages);
      });
      groupDiv.appendChild(showMore);
    }
    container.appendChild(groupDiv);
  });
  if (!window._samplesInitialized) {
    initSampleInteractions();
    window._samplesInitialized = true;
  }
  updateSelectionVisuals();
}
function saveManualOrder() {
  if (!currentJob) return;
  const order = [];
  const visiblePaths = new Set();
  document.querySelectorAll(".sample-group").forEach((group) => {
    const groupKey = group.dataset.group;
    group.querySelectorAll(".sample-card").forEach((card) => {
      order.push({
        path: card.dataset.path,
        group: groupKey,
      });
      visiblePaths.add(card.dataset.path);
    });
  });
  // Images hidden by the per-prompt display limit aren't in the DOM;
  // keep their previously saved placement instead of discarding it.
  (loadManualOrder() || []).forEach((item) => {
    if (!visiblePaths.has(item.path)) order.push(item);
  });
  localStorage.setItem(`sample_order_${currentJob}`, JSON.stringify(order));
}
function loadManualOrder() {
  if (!currentJob) return null;
  const data = localStorage.getItem(`sample_order_${currentJob}`);
  try {
    return data ? JSON.parse(data) : null;
  } catch (e) {
    return null;
  }
}
function createSampleCard(img, container) {
  const card = document.createElement("div");
  card.className = "sample-card";
  card.draggable = true;
  card.dataset.path = img.path;
  card.dataset.name = img.name;
  card.dataset.mtime = img.mtime; // for sorting reference
  // Check selection state
  if (sampleState.selectedPaths.has(img.path)) {
    card.classList.add("selected");
  }
  card.innerHTML = `
        <img src="${img.path}" alt="${escapeHtml(img.name)}" loading="lazy" draggable="false">
        <div class="sample-name">${escapeHtml(img.name)}</div>
        <button class="btn-delete-card" title="Delete Image">🗑</button>
    `;
  // Delete Card Logic
  card.querySelector(".btn-delete-card").addEventListener("click", (e) => {
    e.stopPropagation(); // Don't trigger selection/lightbox
    showConfirm("Delete Image", `Delete "${img.name}"?`, () => {
      deleteSamples([img.path]);
    });
  });
  // Click Selection Logic (Selection + Open Lightbox)
  card.addEventListener("click", (e) => handleSampleClick(e, img, card));
  // Drag Events
  card.addEventListener("dragstart", handleDragStart);
  card.addEventListener("dragover", handleDragOver);
  card.addEventListener("drop", handleDrop);
  card.addEventListener("dragenter", (e) => e.preventDefault());
  container.appendChild(card);
}
// ==========================================
//  Sample Interactions
// ==========================================
// ==========================================
//  Box Selection (Rubber Band)
// ==========================================
let boxSelection = {
  isSelecting: false,
  startX: 0,
  startY: 0,
  element: null,
};
function initSampleInteractions() {
  // Keyboard Navigation
  document.addEventListener("keydown", handleGlobalKeydown);
  // Batch Delete Button
  const btnDelete = $("btn-delete-selected");
  if (btnDelete) {
    btnDelete.addEventListener("click", () => {
      const count = sampleState.selectedPaths.size;
      if (count > 0) {
        showConfirm(
          "Delete Images",
          `Delete ${count} selected image(s)?`,
          () => {
            deleteSamples(Array.from(sampleState.selectedPaths));
          },
        );
      }
    });
  }
  // Box Selection Listeners (on container)
  const container = $("samples-grid"); // This might be hidden initially?
  // We can attach to document or a wrapper.
  // Attaching to 'samples-grid' is safest if it exists.
  if (container) {
    container.addEventListener("mousedown", handleBoxStart);
  }
  document.addEventListener("mousemove", handleBoxMove);
  document.addEventListener("mouseup", handleBoxEnd);
}
function handleBoxStart(e) {
  if (e.target.closest(".sample-card")) return;
  if (e.button !== 0) return;
  boxSelection.isSelecting = true;
  sampleState.isExplicitMultiSelect = true;
  boxSelection.startX = e.pageX;
  boxSelection.startY = e.pageY;
  // Create selection box element
  if (!boxSelection.element) {
    const el = document.createElement("div");
    el.className = "selection-box";
    document.body.appendChild(el);
    boxSelection.element = el;
  }
  const el = boxSelection.element;
  el.style.left = e.pageX + "px";
  el.style.top = e.pageY + "px";
  el.style.width = "0px";
  el.style.height = "0px";
  el.style.display = "block";
  if (!e.ctrlKey && !e.shiftKey) {
    clearSelection();
  }
}
function handleBoxMove(e) {
  if (!boxSelection.isSelecting) return;
  e.preventDefault(); // Stop text selection
  const currentX = e.pageX;
  const currentY = e.pageY;
  const minX = Math.min(boxSelection.startX, currentX);
  const maxX = Math.max(boxSelection.startX, currentX);
  const minY = Math.min(boxSelection.startY, currentY);
  const maxY = Math.max(boxSelection.startY, currentY);
  const el = boxSelection.element;
  el.style.left = minX + "px";
  el.style.top = minY + "px";
  el.style.width = maxX - minX + "px";
  el.style.height = maxY - minY + "px";
  // Update selection in real-time
  updateBoxSelection(minX, minY, maxX, maxY, e.ctrlKey);
}
function handleBoxEnd(e) {
  if (!boxSelection.isSelecting) return;
  boxSelection.isSelecting = false;
  if (boxSelection.element) {
    boxSelection.element.style.display = "none";
  }
}
function updateBoxSelection(x1, y1, x2, y2, isCtrl) {
  const cards = document.querySelectorAll(".sample-card");
  cards.forEach((card) => {
    const rect = card.getBoundingClientRect();
    // Get card coordinates relative to page (since box uses pageX/Y)
    const cardX1 = rect.left + window.scrollX;
    const cardY1 = rect.top + window.scrollY;
    const cardX2 = cardX1 + rect.width;
    const cardY2 = cardY1 + rect.height;
    // Check intersection
    const isOverlapping = !(
      cardX1 > x2 ||
      cardX2 < x1 ||
      cardY1 > y2 ||
      cardY2 < y1
    );
    if (isOverlapping) {
      sampleState.selectedPaths.add(card.dataset.path);
    } else if (!isCtrl) {
      // If not holding Ctrl, box selection is "set" logic, but real-time clearing
      // of things outside box is tricky if we started with a selection.
      // Simplified: Box Selection ADDS to selection during drag.
      // If we want "Select ONLY these", we cleared at start.
      // Scaling back: Standard behavior is Additive if Box touches.
      // To be strict:
      // If we cleared at start, then sampleState contains only what is currently overlapping.
      // But we need to NOT delete things we just added in this drag session if we shrink box.
      // This requires "initialSelection" state. Too complex for raw JS in one function.
      // CURRENT LOGIC: additive only during move.
      // If user shrinks box, items stay selected. (Minor UX quirk but acceptable).
    }
  });
  updateSelectionVisuals();
}
function handleSampleClick(e, img, card) {
  // Lightbox triggers on double click or specific action?
  // User request: "arrow keys to move... even if user is open a specific image"
  // Standard UI: Click = Select, Double Click = Open?
  // Or Click = Open?
  // Plan: Click = Select. Double Click = Lightbox.
  // If modifier keys are used, strictly selection.
  // BUT user said "open a specific image", implying lightbox.
  // Let's implement: Click selects. Double click opens.
  // Also, if you just click and no modifiers, maybe open?
  // "select multiple images then they can drag" -> implies single click might select.
  // Hybrid approach:
  // Simple Click: Selects (and clears others)
  // Ctrl+Click: Toggles
  // Shift+Click: Range
  // Double Click: Open Lightbox
  if (e.ctrlKey || e.metaKey) {
    sampleState.isExplicitMultiSelect = true;
    toggleSelection(img.path);
  } else if (e.shiftKey) {
    sampleState.isExplicitMultiSelect = true;
    selectRange(img.path);
  } else {
    // Simple click: Select and Open Lightbox
    sampleState.isExplicitMultiSelect = false;
    selectSingle(img.path);
    openLightbox(img.path, img.name);
  }
  sampleState.lastSelectedPath = img.path;
}
// Better to attach dblclick to card in createSampleCard, adding it here implies logic change
// Lets add logic in createSampleCard wrapper
// (Modified createSampleCard above needs dblclick listener)
function selectSingle(path) {
  sampleState.selectedPaths.clear();
  sampleState.selectedPaths.add(path);
  updateSelectionVisuals();
}
function toggleSelection(path) {
  if (sampleState.selectedPaths.has(path)) {
    sampleState.selectedPaths.delete(path);
  } else {
    sampleState.selectedPaths.add(path);
  }
  updateSelectionVisuals();
}
function selectRange(targetPath) {
  if (!sampleState.lastSelectedPath) {
    selectSingle(targetPath);
    return;
  }
  // Find indices in the flattened visual list
  // To do this right, we need the current visual DOM order
  const allCards = Array.from(document.querySelectorAll(".sample-card"));
  const startIdx = allCards.findIndex(
    (c) => c.dataset.path === sampleState.lastSelectedPath,
  );
  const endIdx = allCards.findIndex((c) => c.dataset.path === targetPath);
  if (startIdx === -1 || endIdx === -1) return;
  const [min, max] = [Math.min(startIdx, endIdx), Math.max(startIdx, endIdx)];
  // Add range
  // If ctrl not held, clear others? Standard behavior is usually yes for Shift-click
  // But lets keep it additive for now or clear?
  // Windows Explorer: Shift-click clears previous selection (except anchor)
  // Let's clear for simplicity.
  sampleState.selectedPaths.clear();
  for (let i = min; i <= max; i++) {
    sampleState.selectedPaths.add(allCards[i].dataset.path);
  }
  updateSelectionVisuals();
}
function updateSelectionVisuals() {
  const isMultiRoot =
    sampleState.selectedPaths.size > 1 || sampleState.isExplicitMultiSelect;
  const count = sampleState.selectedPaths.size;
  // Batch delete button
  const btnDelete = $("btn-delete-selected");
  if (btnDelete) {
    if (count > 0) {
      btnDelete.classList.remove("hidden");
      btnDelete.textContent = `🗑️ Delete (${count})`;
    } else {
      btnDelete.classList.add("hidden");
    }
  }
  // Toggle multi-select mode on all grids
  document.querySelectorAll(".group-grid").forEach((grid) => {
    grid.classList.toggle("multi-select-mode", isMultiRoot);
  });
  document.querySelectorAll(".sample-card").forEach((card) => {
    if (sampleState.selectedPaths.has(card.dataset.path)) {
      card.classList.add("selected");
    } else {
      card.classList.remove("selected");
    }
  });
}
function clearSelection() {
  sampleState.selectedPaths.clear();
  updateSelectionVisuals();
}
// Drag and Drop Logic
function handleDragStart(e) {
  const path = e.target.closest(".sample-card").dataset.path;
  // If dragging an unselected item, select it first
  if (!sampleState.selectedPaths.has(path)) {
    selectSingle(path);
  }
  e.dataTransfer.setData(
    "text/plain",
    JSON.stringify(Array.from(sampleState.selectedPaths)),
  );
  e.dataTransfer.effectAllowed = "move";
  e.target.closest(".sample-card").classList.add("dragging");
}
function handleDragOver(e) {
  e.preventDefault();
  e.dataTransfer.dropEffect = "move";
  // Remove existing drag indicators
  document
    .querySelectorAll(".drag-over-left, .drag-over-right, .drag-over-grid")
    .forEach((el) => {
      el.classList.remove(
        "drag-over-left",
        "drag-over-right",
        "drag-over-grid",
      );
    });
  const targetCard = e.target.closest(".sample-card");
  const targetGrid = e.target.closest(".group-grid");
  if (targetCard) {
    const rect = targetCard.getBoundingClientRect();
    const relX = e.clientX - rect.left;
    if (relX < rect.width / 2) {
      targetCard.classList.add("drag-over-left");
    } else {
      targetCard.classList.add("drag-over-right");
    }
  } else if (targetGrid) {
    // Visual feedback for dropping into the grid background
    targetGrid.classList.add("drag-over-grid");
  }
}
function handleDrop(e) {
  e.preventDefault();
  document
    .querySelectorAll(".drag-over-left, .drag-over-right, .drag-over-grid")
    .forEach((el) => {
      el.classList.remove(
        "drag-over-left",
        "drag-over-right",
        "drag-over-grid",
      );
    });
  const targetCard = e.target.closest(".sample-card");
  const targetGrid = e.target.closest(".group-grid");
  if (!targetGrid) return;
  try {
    const paths = JSON.parse(e.dataTransfer.getData("text/plain"));
    const allCards = Array.from(document.querySelectorAll(".sample-card"));
    const cardsToMove = allCards.filter((c) => paths.includes(c.dataset.path));
    if (targetCard) {
      const rect = targetCard.getBoundingClientRect();
      const relX = e.clientX - rect.left;
      const insertBefore = relX < rect.width / 2;
      cardsToMove.forEach((card) => {
        if (insertBefore) {
          targetGrid.insertBefore(card, targetCard);
        } else {
          targetGrid.insertBefore(card, targetCard.nextSibling);
        }
      });
    } else {
      // Drop in grid background -> Append to end
      cardsToMove.forEach((card) => {
        targetGrid.appendChild(card);
      });
    }
    cardsToMove.forEach((card) => card.classList.remove("dragging"));
    // Save the new state permanently
    saveManualOrder();
  } catch (err) {
    console.error("Drop error", err);
  }
}
// Keyboard Navigation & Lightbox
function handleGlobalKeydown(e) {
  // Lightbox navigation
  const lightbox = document.querySelector(".lightbox");
  if (lightbox) {
    const currentSrc = lightbox.querySelector("img").getAttribute("src");
    handleLightboxNavigation(e, currentSrc, lightbox);
    return;
  }
  // Grid navigation (Arrow Keys)
  // Only if focus is not in an input
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  if (e.key.startsWith("Arrow")) {
    e.preventDefault();
    navigateGrid(e.key, e.ctrlKey);
  }
  if (e.key === "Enter") {
    const selected = Array.from(sampleState.selectedPaths);
    if (selected.length === 1) {
      const card = document.querySelector(
        `.sample-card[data-path="${selected[0]}"]`,
      );
      openLightbox(selected[0], card ? card.dataset.name : "");
    }
  }
}
// Navigation Helper
function calculateNextIndex(allCards, currentIdx, direction) {
  if (currentIdx === -1) return 0;
  let nextIdx = currentIdx;
  if (direction === "ArrowRight")
    nextIdx = Math.min(currentIdx + 1, allCards.length - 1);
  if (direction === "ArrowLeft") nextIdx = Math.max(currentIdx - 1, 0);
  if (direction === "ArrowUp" || direction === "ArrowDown") {
    const currentRect = allCards[currentIdx].getBoundingClientRect();
    const currentCenter = currentRect.left + currentRect.width / 2;
    const currentY = currentRect.top + currentRect.height / 2;
    let bestDist = Infinity;
    let bestCandidate = -1;
    allCards.forEach((c, i) => {
      if (i === currentIdx) return;
      const r = c.getBoundingClientRect();
      const y = r.top + r.height / 2;
      const x = r.left + r.width / 2;
      // Metric: Minimize vertical dist first, then horizontal.
      const distV = Math.abs(y - currentY);
      const distH = Math.abs(x - currentCenter);
      const score = distV * 2 + distH;
      if (
        (direction === "ArrowUp" && y < currentRect.top) ||
        (direction === "ArrowDown" && y > currentRect.bottom)
      ) {
        if (score < bestDist) {
          bestDist = score;
          bestCandidate = i;
        }
      }
    });
    if (bestCandidate !== -1) nextIdx = bestCandidate;
  }
  return nextIdx;
}
function navigateGrid(direction, isCtrl) {
  // Find current focus (last selected)
  // If no selection, select first
  const allCards = Array.from(document.querySelectorAll(".sample-card"));
  if (allCards.length === 0) return;
  let idx = -1;
  if (sampleState.lastSelectedPath) {
    idx = allCards.findIndex(
      (c) => c.dataset.path === sampleState.lastSelectedPath,
    );
  }
  if (idx === -1) {
    selectSingle(allCards[0].dataset.path);
    allCards[0].scrollIntoView({ block: "center" });
    sampleState.lastSelectedPath = allCards[0].dataset.path;
    return;
  }
  const nextIdx = calculateNextIndex(allCards, idx, direction);
  if (nextIdx !== idx) {
    const path = allCards[nextIdx].dataset.path;
    if (!isCtrl) {
      selectSingle(path);
    } else {
      selectSingle(path);
    }
    allCards[nextIdx].scrollIntoView({ block: "nearest" });
    sampleState.lastSelectedPath = path; // Update visual focus anchor
  }
}
function openLightbox(src, name) {
  // Remove existing
  const existing = document.querySelector(".lightbox");
  if (existing) existing.remove();
  const lb = document.createElement("div");
  lb.className = "lightbox";
  lb.innerHTML = `
        <div class="lightbox-title">${name || ""}</div>
        <img src="${src}">
        <div class="lightbox-metadata hidden"></div>
        <div class="lightbox-nav">
            Use Arrow Keys to navigate | ESC to close
        </div>
    `;
  // Click background to close
  lb.addEventListener("click", (e) => {
    if (e.target === lb) lb.remove();
  });
  document.body.appendChild(lb);
  loadLightboxMetadata(src, lb);
  // Auto-fade navigation hint after 3s
  setTimeout(() => {
    const nav = lb.querySelector(".lightbox-nav");
    if (nav) nav.style.opacity = "0";
  }, 3000);
}
function handleLightboxNavigation(e, currentSrc, lightbox) {
  if (e.key === "Escape") {
    lightbox.remove();
    return;
  }
  if (!e.key.startsWith("Arrow")) return;
  e.preventDefault();
  // Find current index
  const allCards = Array.from(document.querySelectorAll(".sample-card"));
  const idx = allCards.findIndex((c) => c.dataset.path === currentSrc);
  if (idx === -1) return;
  const nextIdx = calculateNextIndex(allCards, idx, e.key);
  if (nextIdx !== idx) {
    const nextCard = allCards[nextIdx];
    const nextPath = nextCard.dataset.path;
    const nextName = nextCard.dataset.name;
    lightbox.querySelector("img").src = nextPath;
    const titleEl = lightbox.querySelector(".lightbox-title");
    if (titleEl) titleEl.textContent = nextName || "";
    // Update metadata
    loadLightboxMetadata(nextPath, lightbox);
    // Also update selection in background
    selectSingle(nextPath);
    nextCard.scrollIntoView({ block: "nearest" });
  }
}
async function loadLightboxMetadata(path, lightbox) {
  const metaEl = lightbox.querySelector(".lightbox-metadata");
  if (!metaEl) return;
  // Convert path from /api/jobs/NAME/samples/... to /api/jobs/NAME/metadata/...
  const metaUrl = path.replace("/samples/", "/metadata/");
  try {
    const res = await fetch(metaUrl);
    if (!res.ok) throw new Error();
    const data = await res.json();
    if (data.parameters) {
      metaEl.textContent = data.parameters;
      metaEl.classList.remove("hidden");
    } else {
      metaEl.classList.add("hidden");
    }
  } catch (e) {
    metaEl.classList.add("hidden");
  }
}
// Add double click listener helper
function addDoubleClick(element, callback) {
  let lastClick = 0;
  element.addEventListener("click", (e) => {
    const now = new Date().getTime();
    if (now - lastClick < 300) {
      callback(e);
    }
    lastClick = now;
  });
}
// ==========================================
//  TensorBoard
// ==========================================
let tbUrl = null;
async function checkTensorBoard() {
  if (!currentJob) return;
  const status = await api(`/api/jobs/${currentJob}/tensorboard/status`);
  updateTbState(status.running, status.url);
}
function updateTbState(running, url) {
  $("btn-tb-launch").classList.toggle("hidden", running);
  $("btn-tb-stop").classList.toggle("hidden", !running);
  $("btn-tb-open").classList.toggle("hidden", !running);
  $("tb-status").textContent = running
    ? `Running on port ${new URL(url).port}`
    : "Not running";
  $("tb-status").style.color = running ? "var(--success)" : "var(--text-muted)";
  if (running && url) {
    tbUrl = url;
    $("tb-placeholder").classList.add("hidden");
    $("tb-iframe").classList.remove("hidden");
    // Only set src if it changed
    if ($("tb-iframe").src !== url) {
      $("tb-iframe").src = url;
    }
  } else {
    tbUrl = null;
    $("tb-placeholder").classList.remove("hidden");
    $("tb-iframe").classList.add("hidden");
    $("tb-iframe").src = "";
  }
}
async function launchTensorBoard() {
  if (!currentJob) return;
  $("btn-tb-launch").disabled = true;
  $("btn-tb-launch").textContent = "Starting...";
  const result = await api(`/api/jobs/${currentJob}/tensorboard`, {
    method: "POST",
  });
  if (result.error) {
    alert(result.error);
    $("btn-tb-launch").disabled = false;
    $("btn-tb-launch").textContent = "\uD83D\uDE80 Launch";
    return;
  }
  // Give TensorBoard a moment to start
  setTimeout(() => {
    updateTbState(true, result.url);
    $("btn-tb-launch").disabled = false;
    $("btn-tb-launch").textContent = "\uD83D\uDE80 Launch";
    showToast("TensorBoard launched");
  }, 2000);
}
async function stopTensorBoard() {
  if (!currentJob) return;
  await api(`/api/jobs/${currentJob}/tensorboard/stop`, { method: "POST" });
  updateTbState(false, null);
  showToast("TensorBoard stopped");
}
// ==========================================
//  Global Settings
// ==========================================
function applyTheme(theme) {
  const t = theme || "github-dark";
  document.documentElement.setAttribute("data-theme", t);
  localStorage.setItem("ui_theme", t);
}
// Build dynamic global settings tabs from architecture registry
function buildGlobalSettingsTabs(registry) {
  const nav = $("global-tabs-nav");
  const content = $("global-tabs-content");
  // Clear old dynamic tabs (keep the static Application tab)
  nav.innerHTML = "";
  content.querySelectorAll(".gtab-pane-dynamic").forEach((el) => el.remove());
  const archs = registry.architectures;
  let isFirst = true;
  for (const [archId, arch] of Object.entries(archs)) {
    // Tab button
    const btn = document.createElement("button");
    btn.className = "tab" + (isFirst ? " active" : "");
    btn.dataset.gtab = archId;
    btn.textContent = arch.display_name + " Models";
    nav.appendChild(btn);
    // Tab pane
    const pane = document.createElement("div");
    pane.id = `gtab-${archId}`;
    pane.className =
      "gtab-pane gtab-pane-dynamic" + (isFirst ? " active" : " hidden");
    for (const [configKey, pathDef] of Object.entries(arch.global_paths)) {
      const group = document.createElement("div");
      group.className = "form-group";
      group.innerHTML = `
                <label>${pathDef.label}</label>
                <input type="text" id="cfg-global-${configKey}" placeholder="${pathDef.placeholder}">
            `;
      pane.appendChild(group);
    }
    // All-in-One sync button
    if (arch.all_in_one && arch.all_in_one_source_key) {
      const syncGroup = document.createElement("div");
      syncGroup.style.marginTop = "8px";
      syncGroup.innerHTML = `
                <button class="btn btn-secondary btn-sm" id="btn-sync-${archId}">\uD83D\uDD04 Use as All-in-One Checkpoint</button>
                <small style="display: block; margin-top: 4px;">Copies the first path to all other fields for this architecture.</small>
            `;
      pane.appendChild(syncGroup);
    }
    // Insert before the static Application tab
    content.insertBefore(pane, $("gtab-app"));
    isFirst = false;
  }
  // Application tab button (always last)
  const appBtn = document.createElement("button");
  appBtn.className = "tab";
  appBtn.dataset.gtab = "app";
  appBtn.textContent = "Application";
  nav.appendChild(appBtn);
  // Bind tab switching
  nav.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      nav.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
      content
        .querySelectorAll(".gtab-pane, .gtab-pane-dynamic")
        .forEach((p) => {
          p.classList.remove("active");
          p.classList.add("hidden");
        });
      tab.classList.add("active");
      const pane = $(`gtab-${tab.dataset.gtab}`);
      if (pane) {
        pane.classList.remove("hidden");
        pane.classList.add("active");
      }
    });
  });
  // Bind all-in-one sync buttons
  for (const [archId, arch] of Object.entries(archs)) {
    if (arch.all_in_one && arch.all_in_one_source_key) {
      const syncBtn = $(`btn-sync-${archId}`);
      if (syncBtn) {
        syncBtn.addEventListener("click", () => {
          const sourceInput = $(`cfg-global-${arch.all_in_one_source_key}`);
          if (sourceInput && sourceInput.value) {
            for (const configKey of Object.keys(arch.global_paths)) {
              $(`cfg-global-${configKey}`).value = sourceInput.value;
            }
            showToast(`${arch.display_name} paths synced!`);
          }
        });
      }
    }
  }
}
async function loadGlobalSettings() {
  // Fetch registry if not cached
  if (!archRegistry) {
    archRegistry = await api("/api/architectures");
    buildGlobalSettingsTabs(archRegistry);
  }
  const config = await api("/api/global-config");
  // Populate path inputs dynamically from registry
  for (const [archId, arch] of Object.entries(archRegistry.architectures)) {
    for (const configKey of Object.keys(arch.global_paths)) {
      const input = $(`cfg-global-${configKey}`);
      if (input) input.value = config.model_paths?.[configKey] || "";
    }
  }
  $("cfg-global-venv").value = config.venv_path || "";
  // Theme
  const theme = config.ui?.theme || "github-dark";
  $("cfg-theme").value = theme;
  applyTheme(theme);
  // Background settings
  const pos = config.ui?.background_position || "50% 50%";
  const dim = config.ui?.dim_level ?? 70;
  const brightness = config.ui?.brightness_level ?? 100;
  const blur = config.ui?.blur_level ?? 10;
  const textShadow = config.ui?.text_shadow_size ?? 0;
  $("cfg-bg-dim").value = dim;
  $("val-bg-dim").textContent = dim + "%";
  $("cfg-bg-brightness").value = brightness;
  $("val-bg-brightness").textContent = brightness + "%";
  $("cfg-bg-blur").value = blur;
  $("val-bg-blur").textContent = blur + "px";
  $("cfg-text-shadow").value = textShadow;
  $("val-text-shadow").textContent = textShadow + "px";
  if (config.ui?.background) {
    applyBackground(
      config.ui.background,
      pos,
      dim,
      brightness,
      blur,
      textShadow,
    );
    $("bg-visual-controls").classList.remove("hidden");
  } else {
    $("bg-pos-group").classList.add("hidden");
    $("bg-visual-controls").classList.add("hidden");
  }
}
async function saveGlobalSettings() {
  // Read existing config first to preserve bg settings
  const existingConfig = await api("/api/global-config");
  // Build model_paths dynamically from registry
  const model_paths = {};
  if (archRegistry) {
    for (const [archId, arch] of Object.entries(archRegistry.architectures)) {
      for (const configKey of Object.keys(arch.global_paths)) {
        const input = $(`cfg-global-${configKey}`);
        if (input) model_paths[configKey] = input.value;
      }
    }
  }
  const config = {
    model_paths,
    venv_path: $("cfg-global-venv").value,
    ui: {
      ...(existingConfig.ui || {}),
      theme: $("cfg-theme").value,
      background_position: `${bgPosPercent.x.toFixed(1)}% ${bgPosPercent.y.toFixed(1)}%`,
      dim_level: parseInt($("cfg-bg-dim").value),
      brightness_level: parseInt($("cfg-bg-brightness").value),
      blur_level: parseInt($("cfg-bg-blur").value),
      text_shadow_size: parseInt($("cfg-text-shadow").value),
    },
  };
  // Apply theme immediately
  applyTheme(config.ui.theme);
  // Live update background if one exists
  if (existingConfig?.ui?.background) {
    applyBackground(
      existingConfig.ui.background,
      config.ui.background_position,
      config.ui.dim_level,
      config.ui.brightness_level,
      config.ui.blur_level,
      config.ui.text_shadow_size,
    );
  }
  await api("/api/global-config", { method: "PUT", body: config });
  closeModal("modal-global-settings");
  showToast("Global settings saved");
}
// === Background Image Functions ===
function applyBackground(
  url,
  position = "50% 50%",
  dim = 70,
  brightness = 100,
  blur = 10,
  textShadow = 0,
) {
  const appContainer = document.querySelector(".app");
  const preview = $("bg-drag-preview");
  const handle = $("bg-drag-handle");
  // Cache for early load
  localStorage.setItem(
    "ui_background",
    JSON.stringify({ url, position, dim, brightness, blur, textShadow }),
  );
  // Remove the early-load style once we have the real container
  const earlyStyle = document.getElementById("early-bg");
  if (earlyStyle) earlyStyle.remove();
  if (url && url !== "none" && url !== "") {
    const root = document.documentElement;
    appContainer.style.backgroundImage = `url('${url}')`;
    appContainer.style.backgroundPosition = position;
    appContainer.classList.add("has-bg");
    root.style.setProperty("--bg-dim", dim / 100);
    root.style.setProperty("--bg-brightness", brightness / 100);
    root.style.setProperty("--bg-blur", blur + "px");
    root.style.setProperty("--text-shadow-size", textShadow + "px");
    preview.style.backgroundImage = `url('${url}')`;
    preview.style.backgroundPosition = position;
    const parts = position.split(" ");
    if (parts.length === 2) {
      bgPosPercent.x = parseFloat(parts[0]);
      bgPosPercent.y = parseFloat(parts[1]);
      handle.style.left = bgPosPercent.x + "%";
      handle.style.top = bgPosPercent.y + "%";
    }
    $("btn-remove-bg").classList.remove("hidden");
    $("bg-pos-group").classList.remove("hidden");
    $("bg-visual-controls").classList.remove("hidden");
  } else {
    appContainer.style.backgroundImage = "none";
    appContainer.classList.remove("has-bg");
    $("btn-remove-bg").classList.add("hidden");
    $("bg-pos-group").classList.add("hidden");
    $("bg-visual-controls").classList.add("hidden");
  }
}
// Drag logic
function updateBgPosFromMouse(e) {
  const container = $("bg-drag-container");
  const rect = container.getBoundingClientRect();
  let x = ((e.clientX - rect.left) / rect.width) * 100;
  let y = ((e.clientY - rect.top) / rect.height) * 100;
  x = Math.max(0, Math.min(100, x));
  y = Math.max(0, Math.min(100, y));
  bgPosPercent = { x, y };
  const posStr = `${x.toFixed(1)}% ${y.toFixed(1)}%`;
  $("bg-drag-handle").style.left = x + "%";
  $("bg-drag-handle").style.top = y + "%";
  $("bg-drag-preview").style.backgroundPosition = posStr;
  document.querySelector(".app").style.backgroundPosition = posStr;
}
$("bg-drag-container").onmousedown = (e) => {
  isDraggingBg = true;
  updateBgPosFromMouse(e);
};
window.addEventListener("mousemove", (e) => {
  if (isDraggingBg) updateBgPosFromMouse(e);
});
window.addEventListener("mouseup", () => {
  isDraggingBg = false;
});
// Upload handler
$("cfg-bg-upload").onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = async (event) => {
    const base64 = event.target.result;
    const res = await api("/api/global/background", {
      method: "POST",
      body: { image: base64 },
    });
    if (res.success) {
      const config = await api("/api/global-config");
      const pos = `${bgPosPercent.x}% ${bgPosPercent.y}%`;
      const dim = parseInt($("cfg-bg-dim").value);
      const brightness = parseInt($("cfg-bg-brightness").value);
      const blur = parseInt($("cfg-bg-blur").value);
      const textShadow = parseInt($("cfg-text-shadow").value);
      applyBackground(res.url, pos, dim, brightness, blur, textShadow);
      // Save to global config
      config.ui = config.ui || {};
      config.ui.background = res.url;
      config.ui.background_position = pos;
      config.ui.dim_level = dim;
      config.ui.brightness_level = brightness;
      config.ui.blur_level = blur;
      config.ui.text_shadow_size = textShadow;
      await api("/api/global-config", { method: "PUT", body: config });
      showToast("Background updated!");
    }
  };
  reader.readAsDataURL(file);
};
// Remove handler
$("btn-remove-bg").onclick = async () => {
  await api("/api/global/background", { method: "DELETE" });
  applyBackground(null);
  const config = await api("/api/global-config");
  if (config.ui) delete config.ui.background;
  await api("/api/global-config", { method: "PUT", body: config });
  showToast("Background removed");
};
// Slider live previews
$("cfg-bg-dim").oninput = (e) => {
  $("val-bg-dim").textContent = e.target.value + "%";
  document.documentElement.style.setProperty("--bg-dim", e.target.value / 100);
};
$("cfg-bg-brightness").oninput = (e) => {
  $("val-bg-brightness").textContent = e.target.value + "%";
  document.documentElement.style.setProperty(
    "--bg-brightness",
    e.target.value / 100,
  );
};
$("cfg-bg-blur").oninput = (e) => {
  $("val-bg-blur").textContent = e.target.value + "px";
  document.documentElement.style.setProperty(
    "--bg-blur",
    e.target.value + "px",
  );
};
$("cfg-text-shadow").oninput = (e) => {
  $("val-text-shadow").textContent = e.target.value + "px";
  document.documentElement.style.setProperty(
    "--text-shadow-size",
    e.target.value + "px",
  );
};
// Theme change handler (live preview)
$("cfg-theme").onchange = (e) => {
  applyTheme(e.target.value);
};
// ==========================================
//  Modals & Helpers
// ==========================================
function openModal(id) {
  $(id).classList.remove("hidden");
}
function closeModal(id) {
  $(id).classList.add("hidden");
}
function showConfirm(title, message, onConfirm) {
  $("confirm-title").textContent = title;
  $("confirm-message").textContent = message;
  const actions = $("confirm-actions");
  actions.innerHTML = "";
  const cancelBtn = document.createElement("button");
  cancelBtn.className = "btn btn-ghost";
  cancelBtn.textContent = "Cancel";
  cancelBtn.onclick = () => closeModal("modal-confirm");
  const confirmBtn = document.createElement("button");
  confirmBtn.className = "btn btn-danger";
  confirmBtn.textContent = "Confirm";
  confirmBtn.onclick = () => {
    closeModal("modal-confirm");
    onConfirm();
  };
  actions.appendChild(cancelBtn);
  actions.appendChild(confirmBtn);
  openModal("modal-confirm");
}
function showToast(msg) {
  const toast = document.createElement("div");
  toast.style.cssText = `
        position: fixed; bottom: 20px; right: 20px; z-index: 300;
        padding: 12px 20px; border-radius: 8px;
        background: var(--bg-tertiary); border: 1px solid var(--border);
        color: var(--text-primary); font-size: 0.9rem;
        box-shadow: var(--shadow); animation: fadeIn 0.2s;
    `;
  toast.textContent = msg;
  document.body.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transition = "opacity 0.3s";
    setTimeout(() => toast.remove(), 300);
  }, 2000);
}
function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
// ==========================================
//  Tabs
// ==========================================
document.querySelectorAll(".tab").forEach((tab) => {
  // don't attach this listener if it's a global tab
  if (tab.closest("#global-tabs-nav")) return;
  tab.addEventListener("click", () => {
    document
      .querySelectorAll(".tab:not(#global-tabs-nav .tab)")
      .forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".tab-pane").forEach((p) => {
      p.classList.remove("active");
      p.classList.add("hidden");
    });
    tab.classList.add("active");
    const pane = $(`tab-${tab.dataset.tab}`);
    pane.classList.remove("hidden");
    pane.classList.add("active");
    localStorage.setItem("lastTab", tab.dataset.tab);
    // Stop polling if switching away from samples (or just reset it)
    if (samplesPollTimer) {
      clearInterval(samplesPollTimer);
      samplesPollTimer = null;
    }
    // Auto-refresh data on tab switch
    if (tab.dataset.tab === "samples") {
      loadSamples();
      samplesPollTimer = setInterval(() => loadSamples(true), 3000);
    }
    if (tab.dataset.tab === "prompts") loadPrompts();
    if (tab.dataset.tab === "tensorboard") checkTensorBoard();
  });
});
// Global tab switching and sync buttons are now handled
// dynamically inside buildGlobalSettingsTabs()
// ==========================================
//  Event Listeners
// ==========================================
$("cfg-enable-sampling").addEventListener("change", (e) => {
  $("group-sample-every").classList.toggle("hidden", !e.target.checked);
});
$("cfg-enable-validation").addEventListener("change", (e) => {
  $("group-validation").classList.toggle("hidden", !e.target.checked);
});
document.querySelectorAll('input[name="duration-unit"]').forEach((el) => {
  el.addEventListener("change", updateDurationUnit);
});
function updateDurationUnit() {
  const unit = document.querySelector(
    'input[name="duration-unit"]:checked',
  ).value;
  const isEpochs = unit === "epochs";
  $("schedule-epochs").classList.toggle("hidden", !isEpochs);
  $("schedule-steps").classList.toggle("hidden", isEpochs);
  $("container-sample-every-epochs").classList.toggle("hidden", !isEpochs);
  $("container-sample-every-steps").classList.toggle("hidden", isEpochs);
  $("container-validate-every-epochs").classList.toggle("hidden", !isEpochs);
  $("container-validate-every-steps").classList.toggle("hidden", isEpochs);
}
// Multiple Datasets
const btnAddDataset = $("btn-add-dataset");
if (btnAddDataset) {
  btnAddDataset.addEventListener("click", () => addSubset(true));
}
// New Job
$("btn-new-job").addEventListener("click", () => {
  $("new-job-name").value = "my_job";
  openModal("modal-new-job");
  $("new-job-name").focus();
});
$("btn-create-job").addEventListener("click", async () => {
  const name = $("new-job-name").value.trim();
  if (!name) return;
  const result = await api("/api/jobs", { method: "POST", body: { name } });
  if (result.error) {
    alert(result.error);
    return;
  }
  closeModal("modal-new-job");
  await loadJobs();
  selectJob(result.name);
  showToast("Job created");
});
// Load GPUs from server
async function loadGPUs() {
  const container = $("cfg-gpu-selection");
  try {
    const gpus = await api("/api/system/gpus");
    container.innerHTML = "";
    if (gpus.length === 0) {
      container.innerHTML =
        "<small>No NVIDIA GPUs detected (CPU only).</small>";
      return;
    }
    gpus.forEach((gpu) => {
      const card = document.createElement("div");
      card.className = "gpu-card selected"; // Default to all selected
      card.dataset.index = gpu.index;
      card.id = `gpu-card-${gpu.index}`;
      card.innerHTML = `
                <div class="gpu-index">GPU ${gpu.index}</div>
                <div class="gpu-name" title="${gpu.name}">${gpu.name}</div>
                <div class="gpu-mem">${gpu.memory}</div>
                <div class="gpu-status">
                    <div class="status-dot"></div>
                    <span class="gpu-status-text">Idle</span>
                </div>
                <input type="checkbox" name="gpu-select" value="${gpu.index}" checked id="gpu-${gpu.index}">
            `;
      card.addEventListener("click", (e) => {
        const checkbox = card.querySelector("input");
        if (e.target.tagName === "INPUT") {
          card.classList.toggle("selected", e.target.checked);
          updateMultiGPUUI();
          checkDirty();
          return;
        }
        checkbox.checked = !checkbox.checked;
        card.classList.toggle("selected", checkbox.checked);
        updateMultiGPUUI();
        checkDirty();
      });
      container.appendChild(card);
    });
    updateGPUActivity();
    updateMultiGPUUI();
  } catch (err) {
    console.error("Failed to load GPUs:", err);
    container.innerHTML = `<small style="color:red">Error: ${err.message}</small>`;
  }
}
// Show the correct mode panel, hide the others.
// Panels use only the "hidden" class for visibility — no disabled-section.
function applyMultiGpuMode(mode) {
  const ddpGroup   = $("group-ddp-opts");
  const fsdpGroup  = $("group-fsdp");
  const fsdp2Group = $("group-fsdp2");
  const dsGroup    = $("group-deepspeed");
  const tpGroup    = $("group-tp-sp");
  if (!ddpGroup || !fsdpGroup || !tpGroup) return;

  ddpGroup.classList.toggle("hidden",   mode !== "ddp");
  fsdpGroup.classList.toggle("hidden",  mode !== "fsdp");
  if (fsdp2Group) fsdp2Group.classList.toggle("hidden", mode !== "fsdp2");
  if (dsGroup) dsGroup.classList.toggle("hidden", mode !== "deepspeed");
  tpGroup.classList.toggle("hidden",    mode !== "tp_sp");

  // Keep hidden checkbox in sync so reconcileFSDPConflicts still works
  const fsdpToggle = $("cfg-use-fsdp");
  if (fsdpToggle) fsdpToggle.checked = (mode === "fsdp" || mode === "fsdp2");

  updateCudaDirectForTpSp();
  updateDeepspeedOffloadUI();
}

function updateCudaDirectForTpSp() {
  const cudaGroup  = $("group-cuda-direct");
  const cudaToggle = $("cfg-use-cuda-direct");
  if (!cudaGroup || !cudaToggle) return;

  const mode   = $("cfg-multigpu-mode")?.value;
  const lockCudaDirect = mode === "tp_sp" || mode === "deepspeed";

  cudaGroup.classList.toggle("disabled-section", lockCudaDirect);
  cudaToggle.disabled = lockCudaDirect;
  if (lockCudaDirect) cudaToggle.checked = false;
}

function updateDeepspeedOffloadUI() {
  const optDevice = $("cfg-ds-offload-optimizer-device");
  const paramDevice = $("cfg-ds-offload-param-device");
  const optNvmeGroup = $("ds-offload-opt-nvme-group");
  const paramNvmeGroup = $("ds-offload-param-nvme-group");
  if (!optDevice || !paramDevice || !optNvmeGroup || !paramNvmeGroup) return;

  optNvmeGroup.classList.toggle("hidden", optDevice.value !== "nvme");
  paramNvmeGroup.classList.toggle("hidden", paramDevice.value !== "nvme");
}

function updateMultiGPUUI() {
  const modeGroup   = $("group-multigpu-mode");
  const cudaGroup   = $("group-cuda-direct");
  const cudaToggle  = $("cfg-use-cuda-direct");
  if (!cudaGroup || !cudaToggle) return;

  const count = document.querySelectorAll('input[name="gpu-select"]:checked').length;

  if (count > 1) {
    if (modeGroup) modeGroup.classList.remove("disabled-section");
    cudaGroup.classList.remove("disabled-section");
    // Keep tp_degree in sync with actual GPU count — the server uses GPU count
    // directly, so this just keeps the display honest.
    const tpDegreeInput = $("cfg-tp-degree");
    if (tpDegreeInput) tpDegreeInput.value = count;
    const tpBackend = $("cfg-tp-backend");
    if (tpBackend && !tpBackend.value) tpBackend.value = "auto";
    const spToggle = $("cfg-sequence-parallel");
    if (spToggle) spToggle.checked = true;
    applyMultiGpuMode($("cfg-multigpu-mode")?.value || "ddp");
  } else {
    // Single GPU — disable mode selector and cuda-direct, hide all panels
    if (modeGroup) modeGroup.classList.add("disabled-section");
    cudaGroup.classList.add("disabled-section");
    cudaToggle.checked = false;
    ["group-ddp-opts", "group-fsdp", "group-fsdp2", "group-deepspeed", "group-tp-sp"].forEach(id => {
      const el = $(id);
      if (el) el.classList.add("hidden");
    });
    const fsdpToggle = $("cfg-use-fsdp");
    if (fsdpToggle) fsdpToggle.checked = false;
  }
}
// Global state for restoring manual offloading values
let lastBlocksValue = "0";
let lastActivationValue = "none";
function reconcileFSDPConflicts() {
  const fsdpToggle = $("cfg-use-fsdp");
  const cudaDirectToggle = $("cfg-use-cuda-direct");
  const blocksInput = $("cfg-blocks-to-swap");
  const activationSelect = $("cfg-activation-offload");
  const blocksGroup = $("group-blocks-to-swap");
  const activationGroup = $("group-activation-offload");
  const strategySelect = $("cfg-fsdp-sharding-strategy");
  if (!fsdpToggle) return;
  if (fsdpToggle.checked) {
    // SAVE CURRENT VALUES before zeroing them (but only if they aren't already 0/none due to a previous toggle)
    if (blocksInput.value !== "0") lastBlocksValue = blocksInput.value;
    if (activationSelect.value !== "none")
      lastActivationValue = activationSelect.value;
    // FORCE TO 0 / NONE
    blocksInput.value = "0";
    activationSelect.value = "none";
    // DISABLE GROUPS
    if (blocksGroup) blocksGroup.classList.add("disabled-section");
    if (activationGroup) activationGroup.classList.add("disabled-section");
  } else {
    // RESTORE PREVIOUS VALUES
    blocksInput.value = lastBlocksValue;
    activationSelect.value = lastActivationValue;
    // ENABLE GROUPS
    if (blocksGroup) blocksGroup.classList.remove("disabled-section");
    if (activationGroup) activationGroup.classList.remove("disabled-section");
  }
  // SHARDING STRATEGY FILTERING (CUDA Direct / Windows compatibility)
  if (strategySelect) {
    const h1 = strategySelect.querySelector('option[value="4"]');
    const h2 = strategySelect.querySelector('option[value="5"]');
    if (cudaDirectToggle && cudaDirectToggle.checked) {
      if (h1) h1.disabled = true;
      if (h2) h2.disabled = true;
      // If current selection was a hybrid one, reset to FULL_SHARD
      if (strategySelect.value === "4" || strategySelect.value === "5") {
        strategySelect.value = "1";
      }
    } else {
      if (h1) h1.disabled = false;
      if (h2) h2.disabled = false;
    }
    // Force update of info box
    if (window.updateFSDPInfoBox) window.updateFSDPInfoBox();
  }
  // TORCH COMPILE vs CUDA DIRECT (mutually exclusive)
  const torchCompileToggle = $("cfg-torch-compile");
  const torchCompileGroup = $("group-torch-compile");
  if (cudaDirectToggle && torchCompileToggle && torchCompileGroup) {
    if (cudaDirectToggle.checked) {
      // Save state before disabling
      if (torchCompileToggle.checked) window._lastTorchCompile = true;
      torchCompileToggle.checked = false;
      torchCompileGroup.classList.add("disabled-section");
    } else {
      torchCompileGroup.classList.remove("disabled-section");
      // Restore previous state if it was saved
      if (window._lastTorchCompile) {
        torchCompileToggle.checked = true;
        window._lastTorchCompile = false;
      }
    }
  }
}
// Global initialization for extra elements
document.addEventListener("DOMContentLoaded", () => {
  const fsdpToggle = $("cfg-use-fsdp");
  const cudaDirectToggle = $("cfg-use-cuda-direct");
  const blocksInput = $("cfg-blocks-to-swap");
  const activationSelect = $("cfg-activation-offload");
  // EXPOSE for usage in other scripts or reactive functions
  window.updateFSDPInfoBox = () => {
    const strategySelect = $("cfg-fsdp-sharding-strategy");
    const fsdpInfo = $("cfg-fsdp-strategy-info");
    if (!strategySelect || !fsdpInfo) return;
    const strategyMap = {
      1: "<strong>FULL_SHARD</strong>: Shards optimizer states, gradients and parameters across all GPUs. Best for maximum VRAM savings.",
      2: "<strong>SHARD_GRAD_OP</strong>: Shards optimizer states and gradients (equivalent to ZeRO-2). Faster than FULL_SHARD but uses more VRAM.",
      3: "<strong>NO_SHARD</strong>: <strong>: Same as DDP</strong> Not recommended for real training",
      4: "<strong>HYBRID_SHARD</strong>: Shards optimizer states, gradients and parameters within each node while each node has a full copy. Use for multi-node setups.",
      5: "<strong>HYBRID_SHARD_ZERO2</strong>: Shards optimizer states and gradients within each node while each node has a full copy.",
    };
    fsdpInfo.innerHTML =
      strategyMap[strategySelect.value] || "Select a strategy to see details.";
  };
  // Multi-GPU mode selector
  const modeSelect = $("cfg-multigpu-mode");
  if (modeSelect) {
    modeSelect.addEventListener("change", (e) => {
      applyMultiGpuMode(e.target.value);
      reconcileFSDPConflicts();
      updateOptimizerOptions();
    });
  }
  const dsOptDevice = $("cfg-ds-offload-optimizer-device");
  const dsParamDevice = $("cfg-ds-offload-param-device");
  if (dsOptDevice) dsOptDevice.addEventListener("change", updateDeepspeedOffloadUI);
  if (dsParamDevice) dsParamDevice.addEventListener("change", updateDeepspeedOffloadUI);
  // fsdpToggle is a hidden input
  if (cudaDirectToggle) {
    cudaDirectToggle.addEventListener("change", reconcileFSDPConflicts);
    // Auto Wrap Policy visibility toggle
    $("cfg-fsdp-auto-wrap-policy").addEventListener("change", (e) => {
      $("fsdp-layer-wrap-group").classList.toggle(
        "hidden",
        e.target.value !== "TRANSFORMER_BASED_WRAP",
      );
      $("fsdp-size-wrap-group").classList.toggle(
        "hidden",
        e.target.value !== "SIZE_BASED_WRAP",
      );
    });
    const fsdp2WrapPolicy = $("cfg-fsdp2-auto-wrap-policy");
    if (fsdp2WrapPolicy) {
      fsdp2WrapPolicy.addEventListener("change", (e) => {
        $("fsdp2-layer-wrap-group").classList.toggle(
          "hidden",
          e.target.value !== "TRANSFORMER_BASED_WRAP",
        );
        $("fsdp2-size-wrap-group").classList.toggle(
          "hidden",
          e.target.value !== "SIZE_BASED_WRAP",
        );
      });
    }
  }
  // Manual value tracking: Update "last known" value when user changes it MANUALLY
  if (blocksInput) {
    blocksInput.addEventListener("change", () => {
      if (!fsdpToggle || !fsdpToggle.checked)
        lastBlocksValue = blocksInput.value;
    });
  }
  if (activationSelect) {
    activationSelect.addEventListener("change", () => {
      if (!fsdpToggle || !fsdpToggle.checked)
        lastActivationValue = activationSelect.value;
    });
  }
  const strategySelect = $("cfg-fsdp-sharding-strategy");
  if (strategySelect) {
    strategySelect.addEventListener("change", window.updateFSDPInfoBox);
    window.updateFSDPInfoBox(); // Initial call
  }
  updateDeepspeedOffloadUI();
  // Initial reconcile call
  reconcileFSDPConflicts();
});
// Load GPUs for generation (separate from training GPU selection)
async function loadGenGPUs() {
  const container = $("gen-gpu-selection");
  if (!container) return;
  try {
    const gpus = await api("/api/system/gpus");
    container.innerHTML = "";
    if (gpus.length === 0) {
      container.innerHTML = "<small>No NVIDIA GPUs detected.</small>";
      return;
    }
    gpus.forEach((gpu, i) => {
      const card = document.createElement("div");
      card.className = "gpu-card" + (i === 0 ? " selected" : "");
      card.dataset.index = gpu.index;
      card.id = `gen-gpu-card-${gpu.index}`;
      card.innerHTML = `
                <div class="gpu-index">GPU ${gpu.index}</div>
                <div class="gpu-name" title="${gpu.name}">${gpu.name}</div>
                <div class="gpu-mem">${gpu.memory}</div>
                <input type="checkbox" name="gen-gpu-select" value="${gpu.index}" ${i === 0 ? "checked" : ""} id="gen-gpu-${gpu.index}">
            `;
      const cb = card.querySelector("input[type=checkbox]");
      card.addEventListener("click", (e) => {
        if (e.target.tagName === "INPUT") {
          card.classList.toggle("selected", e.target.checked);
          updateGenGPULabel();
          return;
        }
        cb.checked = !cb.checked;
        card.classList.toggle("selected", cb.checked);
        updateGenGPULabel();
      });
      container.appendChild(card);
    });
    updateGenGPULabel();
  } catch (err) {
    console.error("Failed to load gen GPUs:", err);
    container.innerHTML = `<small style="color:red">Error: ${err.message}</small>`;
  }
}
function getSelectedGenGPUs() {
  const checked = document.querySelectorAll(
    'input[name="gen-gpu-select"]:checked',
  );
  return Array.from(checked)
    .map((c) => c.value)
    .join(",");
}
function restoreGenGPUSelection(gpuIds) {
  if (!gpuIds) return;
  const ids = gpuIds.split(",").map((s) => s.trim());
  document.querySelectorAll('input[name="gen-gpu-select"]').forEach((cb) => {
    cb.checked = ids.includes(cb.value);
    const card = cb.closest(".gpu-card");
    if (card) card.classList.toggle("selected", cb.checked);
  });
  updateGenGPULabel();
}
function updateGenGPULabel() {
  const label = $("gen-gpu-mode-label");
  const optionsDiv = $("gen-multi-gpu-options");
  if (!label) return;
  const selected = document.querySelectorAll(
    'input[name="gen-gpu-select"]:checked',
  );
  if (selected.length > 1) {
    label.textContent = "— Multi-GPU";
    label.style.color = "var(--success)";
    if (optionsDiv) optionsDiv.style.display = "block";
  } else {
    label.textContent = "";
    label.style.color = "";
    if (optionsDiv) optionsDiv.style.display = "none";
  }
}
async function updateGPUActivity() {
  try {
    const res = await fetch("/api/gpu/activity");
    if (!res.ok) return;
    const activity = await res.json(); // { "0": "training", "1": "sampling" }
    document.querySelectorAll(".gpu-card").forEach((card) => {
      const index = card.dataset.index;
      const status = activity[index] || "idle";
      const textEl = card.querySelector(".gpu-status-text");
      card.classList.remove("active-training", "active-sampling");
      if (status === "training") {
        card.classList.add("active-training");
        textEl.textContent = "Training";
      } else if (status === "sampling") {
        card.classList.add("active-sampling");
        textEl.textContent = "Sampling";
      } else {
        textEl.textContent = "Idle";
      }
    });
  } catch (err) {
    // Silently fail polling
  }
}
$("btn-cancel-job").addEventListener("click", () =>
  closeModal("modal-new-job"),
);
// Enter key in new job name
$("new-job-name").addEventListener("keydown", (e) => {
  if (e.key === "Enter") $("btn-create-job").click();
});
// Save
$("btn-save").addEventListener("click", saveJob);
// Keyboard shortcut: Ctrl+S
document.addEventListener("keydown", (e) => {
  if (e.ctrlKey && e.key === "s") {
    e.preventDefault();
    if (currentJob && isDirty) saveJob();
  }
});
// Clone
$("btn-clone").addEventListener("click", () => {
  if (!currentJob) return;
  // Calculate default name
  let defaultName = `${currentJob}_copy`;
  let counter = 1;
  const uniqueName = (base) => {
    const jobItems = document.querySelectorAll(".job-name");
    for (let item of jobItems) {
      if (item.textContent === base) return false;
    }
    return true;
  };
  while (!uniqueName(defaultName)) {
    counter++;
    defaultName = `${currentJob}_copy_${counter}`;
  }
  // Open Modal
  $("clone-job-name").value = defaultName;
  openModal("modal-clone-job");
  $("clone-job-name").focus();
  $("clone-job-name").select();
});
// Confirm Clone
$("btn-confirm-clone").addEventListener("click", async () => {
  const newName = $("clone-job-name").value.trim();
  if (!newName) return;
  const result = await api(`/api/jobs/${currentJob}/clone`, {
    method: "POST",
    body: { newName: newName },
  });
  if (result.error) {
    alert(result.error);
    return;
  }
  closeModal("modal-clone-job");
  await loadJobs();
  selectJob(result.name);
  showToast("Job cloned");
});
// Cancel Clone
$("btn-cancel-clone").addEventListener("click", () =>
  closeModal("modal-clone-job"),
);
// Enter key in clone job name
$("clone-job-name").addEventListener("keydown", (e) => {
  if (e.key === "Enter") $("btn-confirm-clone").click();
});
// Delete
$("btn-delete").addEventListener("click", () => {
  if (!currentJob) return;
  showConfirm(
    "Delete Job",
    `Delete "${currentJob}" and all its files? This cannot be undone.`,
    async () => {
      const deletedJob = currentJob;
      await api(`/api/jobs/${deletedJob}`, { method: "DELETE" });
      // Clean up all localStorage keys for the deleted job
      localStorage.removeItem(`prompt_transient_${deletedJob}`);
      localStorage.removeItem(`sample_order_${deletedJob}`);
      localStorage.removeItem("lastJob");
      currentJob = null;
      isDirty = false;
      $("btn-save").classList.add("hidden");
      $("btn-discard").classList.add("hidden");
      emptyState.classList.remove("hidden");
      jobEditor.classList.add("hidden");
      await loadJobs();
      showToast("Job deleted");
    },
  );
});
// Train
$("btn-run").addEventListener("click", async () => {
  if (!currentJob) return;
  let warningMsg = "";
  // Check sampling Logic
  if ($("cfg-enable-sampling").checked && currentPrompts.length === 0) {
    warningMsg =
      "Sampling is enabled but no prompts are defined.\n\nContinue training without generating samples...\n\n";
  }
  // Auto-save first
  if (isDirty) await saveJob();
  const result = await api(`/api/jobs/${currentJob}/train/start`, {
    method: "POST",
  });
  if (result.error) {
    alert(result.error);
    return;
  }
  updateRunningState(true);
  resetConsole();
  if (warningMsg) appendConsole(warningMsg);
  // Auto-switch to console tab
  document.querySelector('[data-tab="console"]').click();
  showToast("Training started");
});
// Generate
$("btn-gen-sample").addEventListener("click", async () => {
  if (!currentJob) return;
  savePromptTransientSettings();
  if (isDirty) await saveJob();
  if (currentPrompts.length === 0) {
    showToast("Add sample prompts first");
    return;
  }
  const payload = {};
  const loraPath = $("gen-lora-select").value;
  if (loraPath) {
    payload.network_weights = loraPath;
    payload.network_mul = parseFloat($("gen-lora-mul").value) || 1.0;
  }
  // Add Anima generation params
  payload.flow_shift = parseFloat($("cfg-flow-shift").value) || 3.0;
  payload.flash_attn = $("gen-flash-attn").checked;
  payload.sage_attn = $("gen-sage-attn").checked;
  payload.gen_gpu_ids = getSelectedGenGPUs();
  payload.gen_multi_gpu_mode = $("gen-multi-gpu-mode").value;
  const result = await api(`/api/jobs/${currentJob}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: Object.assign(payload, {
      keep_loaded: $("chk-keep-loaded").checked,
    }),
  });
  if (result.error) {
    alert(result.error);
    return;
  }
  appendConsole(
    `Starting generation...\n${loraPath ? `Using LoRA: ${loraPath} (x${payload.network_mul})` : "(Using base model)"}\nFlow Shift: ${payload.flow_shift}\n\n`,
  );
  showToast("Generation started");
});
// Unload Model
$("btn-unload-model").addEventListener("click", async () => {
  if (!currentJob) return;
  showToast("Unloading model...");
  const result = await api(`/api/jobs/${currentJob}/unload`, {
    method: "POST",
  });
  if (result.success) {
    showToast(result.message || "Model unloaded");
  } else {
    alert(result.error);
  }
});
$("btn-refresh-checkpoints").addEventListener("click", () => {
  loadCheckpoints();
  showToast("Checkpoints refreshed");
});
// Stop
$("btn-stop").addEventListener("click", () => {
  if (!currentJob) return;
  showConfirm(
    "Stop Training",
    `Stop training for "${currentJob}"?`,
    async () => {
      await api(`/api/jobs/${currentJob}/train/stop`, { method: "POST" });
      updateRunningState(false);
      showToast("Training stopped");
    },
  );
});
// Console clear
$("btn-clear-console").addEventListener("click", () => {
  resetConsole();
});
// Samples refresh
$("btn-refresh-samples").addEventListener("click", loadSamples);
const samplesLimitSelect = $("samples-limit");
if (samplesLimitSelect) {
  samplesLimitSelect.value = getSamplesLimit();
  samplesLimitSelect.addEventListener("change", () => {
    setSamplesLimit(samplesLimitSelect.value);
    sampleState.expandedGroups.clear();
    if (sampleState.allImages.length > 0) {
      renderSampleGroups(sampleState.allImages);
    }
  });
}
// TensorBoard
$("btn-tb-launch").addEventListener("click", launchTensorBoard);
$("btn-tb-stop").addEventListener("click", () => {
  showConfirm(
    "Stop TensorBoard",
    "Stop the TensorBoard server for this job?",
    stopTensorBoard,
  );
});
$("btn-tb-open").addEventListener("click", () => {
  if (tbUrl) window.open(tbUrl, "_blank");
});
// Global Settings
$("btn-global-settings").addEventListener("click", () => {
  loadGlobalSettings();
  openModal("modal-global-settings");
});
$("btn-close-global").addEventListener("click", () =>
  closeModal("modal-global-settings"),
);
$("btn-save-global").addEventListener("click", saveGlobalSettings);
// Prompts
$("btn-add-prompt").addEventListener("click", addPrompt);
$("btn-apply-global").addEventListener("click", applyGlobalSettings);
// Persistence for Prompt Tab settings
[
  "gen-lora-select",
  "gen-lora-mul",
  "chk-keep-loaded",
  "gen-flash-attn",
  "gen-multi-gpu-mode",
  "global-w",
  "global-h",
  "global-s",
  "global-l",
  "global-d",
].forEach((id) => {
  $(id).addEventListener("change", savePromptTransientSettings);
  if ($(id).tagName === "INPUT") {
    $(id).addEventListener("input", savePromptTransientSettings);
  }
});
// Job Settings
$("btn-open-folder").addEventListener("click", async () => {
  if (!currentJob) return;
  await api(`/api/jobs/${currentJob}/open-folder`, { method: "POST" });
});
$("btn-clear-logs").addEventListener("click", () => {
  if (!currentJob) return;
  showConfirm(
    "Clear Logs",
    "Delete all TensorBoard logs for this job?",
    async () => {
      await api(`/api/jobs/${currentJob}/clear-logs`, { method: "POST" });
      showToast("Logs cleared");
    },
  );
});
$("btn-reset-config").addEventListener("click", () => {
  if (!currentJob) return;
  showConfirm(
    "Reset Config",
    "Reset all settings to template defaults?",
    async () => {
      await api(`/api/jobs/${currentJob}/reset-config`, { method: "POST" });
      selectJob(currentJob);
      showToast("Config reset to defaults");
    },
  );
});
// Close modals on backdrop click
document.querySelectorAll(".modal").forEach((modal) => {
  modal.addEventListener("click", (e) => {
    if (e.target === modal) {
      modal.classList.add("hidden");
    }
  });
});
// ==========================================
//  Progressive Resolution Schedule
// ==========================================
function renderProgressivePhases() {
  const resList = ($("cfg-resolution").value || "")
    .split(",").map(r => parseInt(r.trim())).filter(r => r > 0);
  const container = $("progressive-reso-phases");
  if (!container) return;

  // Preserve existing fraction values by index before clearing
  const existing = Array.from(container.querySelectorAll(".prog-reso-frac"))
    .map(el => parseFloat(el.value) || 0);

  container.innerHTML = "";

  if (resList.length < 2) {
    container.innerHTML = '<small>Enter at least 2 resolutions above to configure phases.</small>';
    updateProgressiveSum();
    return;
  }

  const defaultFrac = +(1 / resList.length).toFixed(2);

  // All phases go in a single form-row so they appear side-by-side
  const row = document.createElement("div");
  row.className = "form-row";

  resList.forEach((r, i) => {
    const group = document.createElement("div");
    group.className = "form-group";

    const label = document.createElement("label");
    label.textContent = `${r}px`;

    const input = document.createElement("input");
    input.type = "number";
    input.className = "prog-reso-frac";
    input.min = "0.01";
    input.max = "0.99";
    input.step = "0.01";
    input.value = (existing[i] !== undefined && existing[i] > 0)
      ? existing[i].toFixed(2) : defaultFrac.toFixed(2);

    const hint = document.createElement("small");
    hint.textContent = `${Math.round(parseFloat(input.value) * 100)}% of steps`;

    input.addEventListener("input", () => {
      hint.textContent = `${Math.round(parseFloat(input.value) * 100)}% of steps`;
      updateProgressiveSum();
    });

    group.appendChild(label);
    group.appendChild(input);
    group.appendChild(hint);
    row.appendChild(group);
  });

  container.appendChild(row);

  // Restore fractions when loading from a saved config
  if (window._pendingProgressiveSchedule) {
    const fracs = window._pendingProgressiveSchedule
      .split(",").map(p => parseFloat(p.split(":")[1]) || 0);
    container.querySelectorAll(".prog-reso-frac").forEach((inp, i) => {
      if (fracs[i] !== undefined) {
        inp.value = fracs[i].toFixed(2);
        inp.dispatchEvent(new Event("input"));
      }
    });
    window._pendingProgressiveSchedule = null;
  }

  updateProgressiveSum();
}

function updateProgressiveSum() {
  const inputs = document.querySelectorAll(".prog-reso-frac");
  const sum = Array.from(inputs).reduce((acc, el) => acc + (parseFloat(el.value) || 0), 0);
  const hint = $("progressive-reso-sum-hint");
  if (!hint) return;
  const ok = Math.abs(sum - 1.0) < 0.015;
  const sumStr = `Sum: ${sum.toFixed(2)}`;
  // Show sum inline in the hint text with colour
  hint.innerHTML = `Each fraction is the portion of total steps for that resolution. Must sum to 1.0. &nbsp;<span style="font-weight:600;color:${ok ? "var(--success,#4caf50)" : "var(--error,#f44336)"}">${sumStr}</span>`;
}

// Toggle panel visibility and re-render phases
document.addEventListener("change", (e) => {
  if (e.target.id === "cfg-progressive-reso") {
    const panel = $("progressive-reso-panel");
    if (e.target.checked) {
      panel.classList.remove("hidden");
      renderProgressivePhases();
    } else {
      panel.classList.add("hidden");
    }
  }
});

// Re-render phases when the resolution list changes
document.addEventListener("input", (e) => {
  if (e.target.id === "cfg-resolution" && $("cfg-progressive-reso")?.checked) {
    renderProgressivePhases();
  }
});

// ==========================================
//  Init
// ==========================================
async function init() {
  // 1. FAST LOAD: Apply cached visual settings immediately (Flicker prevention)
  const cachedTheme = localStorage.getItem("ui_theme");
  if (cachedTheme) applyTheme(cachedTheme);
  const cachedBg = localStorage.getItem("ui_background");
  if (cachedBg) {
    try {
      const bg = JSON.parse(cachedBg);
      applyBackground(
        bg.url,
        bg.position,
        bg.dim,
        bg.brightness,
        bg.blur,
        bg.textShadow,
      );
    } catch (e) { }
  }
  // 2. Normal Init
  connectWS();
  await loadJobs();
  // Start status polling
  setInterval(updateGPUActivity, 3000);
  // Watch for config changes
  document.addEventListener("input", (e) => {
    if (e.target.id && e.target.id.startsWith("cfg-")) {
      checkDirty();
    }
  });
  document.addEventListener("change", (e) => {
    if (e.target.id && e.target.id.startsWith("cfg-")) {
      checkDirty();
    }
  });
  // Optimizer custom bindings
  $("cfg-optimizer").addEventListener("change", updateOptimizerOptions);
  $("cfg-lr-scheduler").addEventListener("change", updateLrSchedulerOptions);
  // Activation offload <-> blocks to swap mutual exclusivity
  $("cfg-activation-offload").addEventListener(
    "change",
    updateActivationOffloadUI,
  );
  // Discard Button
  $("btn-discard").addEventListener("click", discardChanges);
  // Mutual exclusivity for Flash/Sage Attention
  const enforceMutualAttention = (flashId, sageId) => {
    const flash = $(flashId);
    const sage = $(sageId);
    if (!flash || !sage) return;
    flash.addEventListener("change", () => {
      if (flash.checked) sage.checked = false;
      if (flashId.startsWith("gen-")) savePromptTransientSettings();
    });
    sage.addEventListener("change", () => {
      if (sage.checked) flash.checked = false;
      if (flashId.startsWith("gen-")) savePromptTransientSettings();
    });
  };
  enforceMutualAttention("gen-flash-attn", "gen-sage-attn");
  // Restore Job
  const lastJob = localStorage.getItem("lastJob");
  if (lastJob) {
    const jobExists = Array.from(
      document.querySelectorAll(".job-item .job-name"),
    ).some((el) => el.textContent === lastJob);
    if (jobExists) {
      await selectJob(lastJob);
    } else {
      localStorage.removeItem("lastJob");
    }
  }
  // Restore Tab
  const lastTab = localStorage.getItem("lastTab");
  if (lastTab && currentJob) {
    const tabEl = document.querySelector(`.tab[data-tab="${lastTab}"]`);
    if (tabEl) tabEl.click();
  }
  // 3. Sync Settings: Load from server and refresh cache
  const globalConfig = await api("/api/global-config");
  if (globalConfig?.ui?.theme) {
    applyTheme(globalConfig.ui.theme);
  }
  // Apply saved background
  if (globalConfig?.ui?.background) {
    applyBackground(
      globalConfig.ui.background,
      globalConfig.ui.background_position || "50% 50%",
      globalConfig.ui.dim_level ?? 70,
      globalConfig.ui.brightness_level ?? 100,
      globalConfig.ui.blur_level ?? 10,
      globalConfig.ui.text_shadow_size ?? 0,
    );
  } else {
    applyBackground("none");
  }
}
init();
window.addEventListener("beforeunload", () => savePromptTransientSettings());
