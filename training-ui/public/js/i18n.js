(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = { createI18n: factory };
  } else {
    root.animaI18n = factory({
      storage: root.localStorage,
      navigator: root.navigator,
      document: root.document,
      CustomEvent: root.CustomEvent,
    });
  }
})(typeof window !== "undefined" ? window : globalThis, function createI18n(runtime = {}) {
  const storage = runtime.storage || { getItem: () => null, setItem: () => {} };
  const navigator = runtime.navigator || { language: "en" };
  const document = runtime.document || null;
  const CustomEvent = runtime.CustomEvent;
  const STORAGE_KEY = "ui_locale";

  const translations = {
    en: {
      "jobs.empty": "No jobs yet",
      "document.title": "Anima LoRA Training",
      "samples.showMore": "Show all ({count} more)",
      "prompts.placeholder": "Enter prompt text...",
      "actions.delete": "Delete",
      "actions.train": "▶ Train",
      "actions.refresh": "🔄 Refresh",
      "language.switchToChinese": "中文",
      "language.switchToEnglish": "English",
      "hardware.system": "SYSTEM",
      "hardware.core": "CORE",
      "hardware.ram": "RAM",
      "hardware.vram": "VRAM",
      "hardware.power": "POWER",
      "hardware.monitor": "Hardware Monitor",
      "hardware.loading": "Connecting...",
      "hardware.idle": "Idle",
      "hardware.training": "Training",
      "hardware.sampling": "Sampling",
      "hardware.multiGpu": "Multi-GPU",
      "jobs.unsavedChanges": "Unsaved changes. Switch anyway?",
      "jobs.loadFailed": "Failed to load job: {message}",
      "jobs.created": "Job created",
      "jobs.cloned": "Job cloned",
      "jobs.deleted": "Job deleted",
      "jobs.saved": "Job saved",
      "jobs.discardChanges": "Discard Changes",
      "jobs.discardMessage": "Discard all unsaved changes and revert to last saved state?",
      "jobs.changesDiscarded": "Changes discarded",
      "jobs.deleteTitle": "Delete Job",
      "jobs.deleteMessage": 'Delete "{job}" and all its files? This cannot be undone.',
      "jobs.stopTitle": "Stop Training",
      "jobs.stopMessage": 'Stop training for "{job}"?',
      "training.started": "Training started",
      "training.stopped": "Training stopped",
      "training.waiting": "Waiting for training to start...",
      "training.noPromptsWarning": "Sampling is enabled but no prompts are defined.\n\nContinue training without generating samples...\n\n",
      "generation.started": "Generation started",
      "generation.addPromptsFirst": "Add sample prompts first",
      "generation.unloading": "Unloading model...",
      "generation.unloaded": "Model unloaded",
      "generation.checkpointsRefreshed": "Checkpoints refreshed",
      "generation.starting": "Starting generation...",
      "generation.usingLora": "Using LoRA: {path} (x{strength})",
      "generation.usingBase": "(Using base model)",
      "generation.flowShift": "Flow Shift: {value}",
      "console.clear": "Clear",
      "console.openLightboxHint": "Use Arrow Keys to navigate | ESC to close",
      "tensorboard.starting": "Starting...",
      "tensorboard.launch": "Launch",
      "tensorboard.launched": "TensorBoard launched",
      "tensorboard.stopped": "TensorBoard stopped",
      "tensorboard.runningOnPort": "Running on port {port}",
      "tensorboard.notRunning": "Not running",
      "tensorboard.stopTitle": "Stop TensorBoard",
      "tensorboard.stopMessage": "Stop the TensorBoard server for this job?",
      "global.settingsSaved": "Global settings saved",
      "global.backgroundUpdated": "Background updated!",
      "global.backgroundRemoved": "Background removed",
      "global.pathsSynced": "{name} paths synced!",
      "samples.uncategorized": "Uncategorized",
      "samples.deleteTitle": "Delete Image",
      "samples.deleteMessage": 'Delete "{name}"?',
      "samples.deleteSelectedTitle": "Delete Images",
      "samples.deleteSelectedMessage": "Delete {count} selected image(s)?",
      "samples.deleteSelectedButton": "🗑️ Delete ({count})",
      "dataset.emptyPath": "Empty Path",
      "dataset.enterPathFirst": "Please enter a directory path first",
      "dataset.error": "Error: {message}",
      "dataset.duplicatePaths": "Error: Duplicate Image Directories detected. Each subset must have a unique path.",
      "dataset.imageDirectory": "Image Directory",
      "dataset.openFolder": "Open folder",
      "dataset.numRepeats": "Num Repeats",
      "dataset.keepTokens": "Keep Tokens",
      "dataset.captionPrefix": "Caption Prefix",
      "dataset.captionDropout": "Caption Dropout Rate",
      "dataset.tagDropout": "Tag Dropout Rate",
      "dataset.dropoutEvery": "Dropout Every N Epochs",
      "dataset.shuffleCaptions": "Shuffle Captions",
      "dataset.flipAugmentations": "Flip Augmentations",
      "dataset.cacheMetadata": "Cache Metadata",
      "dataset.regularization": "Regularization Dataset",
      "dataset.title": "Dataset",
      "dataset.disabledZero": "0 = disabled",
      "dataset.regularizationHint": "Images in this folder are used as regularization (class images) to prevent overfitting.",
      "prompts.skip": "Skip",
      "prompts.randomSeedsApplied": "Random seeds applied to all prompts",
      "prompts.appliedToAll": "Global settings applied to all prompts",
      "prompts.baseModel": "Base Model (No LoRA)",
      "samples.prompt": "Prompt",
      "samples.showPerPrompt": "Show per prompt",
      "samples.recent": "{count} most recent",
      "samples.all": "All",
      "config.resetTitle": "Reset Config",
      "config.resetMessage": "Reset all settings to template defaults?",
      "config.resetDone": "Config reset to defaults",
      "settings.clearLogsTitle": "Clear Logs",
      "settings.clearLogsMessage": "Delete all TensorBoard logs for this job?",
      "settings.logsCleared": "Logs cleared",
      "confirm.cancel": "Cancel",
      "confirm.confirm": "Confirm",
      "gpu.noNvidia": "No NVIDIA GPUs detected (CPU only).",
      "gpu.noNvidiaGeneration": "No NVIDIA GPUs detected.",
      "gpu.error": "Error: {message}",
      "progressive.enterTwo": "Enter at least 2 resolutions above to configure phases.",
      "progressive.percentOfSteps": "{percent}% of steps",
      "progressive.sumHint": "Each fraction is the portion of total steps for that resolution. Must sum to 1.0.",
      "progressive.sum": "Sum: {sum}",
      "global.modelsSuffix": " Models",
      "global.application": "Application",
      "global.settings": "⚙️ Global Settings",
      "global.path.dit_path": "DiT Model Path",
      "global.path.qwen3_path": "Qwen3 Text Encoder Path",
      "global.path.vae_path": "VAE Path",
      "global.path.lumina_dit_path": "Lumina DiT Model Path",
      "global.path.gemma2_path": "Gemma2 Model Path",
      "global.path.lumina_vae_path": "Lumina VAE Path",
      "global.allInOne": "Use as All-in-One Checkpoint",
      "global.allInOneHint": "Copies the first path to all other fields for this architecture.",
      "optimizer.muonWarning": "Muon is incompatible with DeepSpeed and FSDP1. Reverted Multi-GPU mode to DDP.",
      "optimizer.muonLoraWarning": "Muon optimizer is not supported for LoRA training. Reverted to AdamW8bit.",
    },
    "zh-CN": {
      text: {
        "Jobs": "任务",
        "🎯 Jobs": "🎯 任务",
        "+ New": "+ 新建",
        "Global Settings": "全局设置",
        "⚙️ Global Settings": "⚙️ 全局设置",
        "No Job Selected": "未选择任务",
        "Create a new training job or select one from the sidebar": "创建新的训练任务，或从侧边栏选择一个任务",
        "Job Name": "任务名称",
        "Save": "保存",
        "Discard": "放弃",
        "Clone": "克隆",
        "Train": "训练",
        "Stop": "停止",
        "Training": "训练",
        "Dataset": "数据集",
        "Network": "网络",
        "Multi-GPUs": "多 GPU",
        "Prompts": "提示词",
        "Samples": "样本",
        "Console": "控制台",
        "Optimization": "优化",
        "Learning Rate": "学习率",
        "Text Encoder LR": "文本编码器学习率",
        "Optimizer": "优化器",
        "LR Scheduler": "学习率调度器",
        "LR Warmup Steps": "学习率预热步数",
        "Weight Decay": "权重衰减",
        "Seed": "随机种子",
        "Restart Cycles": "重启周期",
        "Min LR Ratio": "最小学习率比例",
        "Training Schedule": "训练计划",
        "Duration Unit": "时长单位",
        "Epochs": "轮数",
        "Steps": "步数",
        "Max Epochs": "最大轮数",
        "Save Every N Epochs": "每 N 轮保存",
        "Keep Last N Epochs": "保留最近 N 轮",
        "Max Steps": "最大步数",
        "Save Every N Steps": "每 N 步保存",
        "Keep Last N Steps": "保留最近 N 步",
        "Output Name": "输出名称",
        "Save Format": "保存格式",
        "Performance": "性能",
        "Mixed Precision": "混合精度",
        "Save Precision": "保存精度",
        "DataLoader Workers": "DataLoader 工作进程",
        "Caching": "缓存",
        "Dataset Folders": "数据集文件夹",
        "+ Add Dataset": "+ 添加数据集",
        "Global Dataset Settings": "全局数据集设置",
        "Resolution(s)": "分辨率",
        "Batch Size(s)": "批大小",
        "Gradient Accumulation": "梯度累积",
        "Caption Extension": "标题扩展名",
        "Bucketing": "分桶",
        "Validation": "验证",
        "Network Module": "网络模块",
        "Network Dim (Rank)": "网络维度（Rank）",
        "Network Alpha": "网络 Alpha",
        "Network Dropout": "网络 Dropout",
        "Resume Training": "恢复训练",
        "HuggingFace": "HuggingFace",
        "In-Training Sampling": "训练中采样",
        "Test Generation": "测试生成",
        "Base Model (No LoRA)": "基础模型（无 LoRA）",
        "LoRA Strength": "LoRA 强度",
        "GPU Selection": "GPU 选择",
        "Generate": "生成",
        "+ Add Prompt": "+ 添加提示词",
        "Sample Prompts": "样本提示词",
        "Negative Prompt": "负面提示词",
        "Apply to All": "应用到全部",
        "Hardware Allocation": "硬件分配",
        "Multi-GPU Optimization": "多 GPU 优化",
        "Diagnostics": "诊断",
        "Generated Samples": "生成的样本",
        "Show per prompt": "每个提示词显示",
        "most recent": "最近的",
        "All": "全部",
        "Delete Selected": "删除选中项",
        "Delete Image": "删除图像",
        "Refresh": "刷新",
        "No sample images yet. They will appear here during training.": "暂无样本图像。训练期间生成的图像会显示在这里。",
        "Training Console": "训练控制台",
        "TensorBoard": "TensorBoard",
        "Launch": "启动",
        "Open": "打开",
        "Job Maintenance": "任务维护",
        "Open Job Folder": "打开任务文件夹",
        "Logging": "日志",
        "Clear TensorBoard Logs": "清除 TensorBoard 日志",
        "Danger Zone": "危险区域",
        "Reset Config to Defaults": "将配置重置为默认值",
        "Create New Training Job": "创建新的训练任务",
        "Cancel": "取消",
        "Create": "创建",
        "New Job Name": "新任务名称",
        "Confirm": "确认",
        "Are you sure?": "确定要继续吗？",
        "Theme": "主题",
        "Background Image": "背景图像",
        "Choose Image": "选择图像",
        "Remove": "移除",
        "Positioning (Drag to focal point)": "位置（拖动到焦点）",
        "Dim Level (Overlay Opacity): ": "变暗程度（覆盖层不透明度）：",
        "Image Brightness: ": "图像亮度：",
        "Backdrop Blur: ": "背景模糊：",
        "Text Shadow Intensity: ": "文字阴影强度：",
        "Application": "应用程序",
        "Save Settings": "保存设置",
        "Private": "私有",
        "Public": "公开",
        "Auto": "自动",
        "None": "无",
        "CPU": "CPU",
        "GPU": "GPU",
        "Core": "核心",
        "Power": "功耗",
        "Training": "训练中",
        "Sampling": "采样中",
        "Idle": "空闲",
      },
      "jobs.empty": "暂无任务",
      "document.title": "Anima LoRA 训练",
      "samples.showMore": "显示全部（另外 {count} 个）",
      "prompts.placeholder": "输入提示词文本...",
      "actions.delete": "删除",
      "actions.train": "▶ 训练",
      "actions.refresh": "🔄 刷新",
      "language.switchToChinese": "中文",
      "language.switchToEnglish": "English",
      "hardware.system": "系统",
      "hardware.core": "核心",
      "hardware.ram": "内存",
      "hardware.vram": "显存",
      "hardware.power": "功耗",
      "hardware.monitor": "硬件监控",
      "hardware.loading": "连接中...",
      "hardware.idle": "空闲",
      "hardware.training": "训练中",
      "hardware.sampling": "采样中",
      "hardware.multiGpu": "多 GPU",
      "jobs.unsavedChanges": "有未保存的更改。仍要切换吗？",
      "jobs.loadFailed": "加载任务失败：{message}",
      "jobs.created": "任务已创建",
      "jobs.cloned": "任务已克隆",
      "jobs.deleted": "任务已删除",
      "jobs.saved": "任务已保存",
      "jobs.discardChanges": "放弃更改",
      "jobs.discardMessage": "放弃所有未保存的更改并恢复到上次保存的状态？",
      "jobs.changesDiscarded": "已放弃更改",
      "jobs.deleteTitle": "删除任务",
      "jobs.deleteMessage": "删除“{job}”及其所有文件？此操作无法撤销。",
      "jobs.stopTitle": "停止训练",
      "jobs.stopMessage": "停止“{job}”的训练？",
      "training.started": "训练已开始",
      "training.stopped": "训练已停止",
      "training.waiting": "等待训练开始...",
      "training.noPromptsWarning": "已启用采样，但尚未定义提示词。\n\n仍要继续训练且不生成样本吗？\n\n",
      "generation.started": "生成已开始",
      "generation.addPromptsFirst": "请先添加样本提示词",
      "generation.unloading": "正在卸载模型...",
      "generation.unloaded": "模型已卸载",
      "generation.checkpointsRefreshed": "检查点已刷新",
      "generation.starting": "开始生成...",
      "generation.usingLora": "使用 LoRA：{path}（强度 {strength}）",
      "generation.usingBase": "（使用基础模型）",
      "generation.flowShift": "流偏移：{value}",
      "console.clear": "清空",
      "console.openLightboxHint": "使用方向键导航 | ESC 关闭",
      "tensorboard.starting": "启动中...",
      "tensorboard.launch": "启动",
      "tensorboard.launched": "TensorBoard 已启动",
      "tensorboard.stopped": "TensorBoard 已停止",
      "tensorboard.runningOnPort": "运行于端口 {port}",
      "tensorboard.notRunning": "未运行",
      "tensorboard.stopTitle": "停止 TensorBoard",
      "tensorboard.stopMessage": "停止此任务的 TensorBoard 服务？",
      "global.settingsSaved": "全局设置已保存",
      "global.backgroundUpdated": "背景已更新！",
      "global.backgroundRemoved": "背景已移除",
      "global.pathsSynced": "{name} 路径已同步！",
      "samples.uncategorized": "未分类",
      "samples.deleteTitle": "删除图像",
      "samples.deleteMessage": "删除“{name}”？",
      "samples.deleteSelectedTitle": "删除图像",
      "samples.deleteSelectedMessage": "删除选中的 {count} 张图像？",
      "samples.deleteSelectedButton": "🗑️ 删除（{count}）",
      "dataset.emptyPath": "空路径",
      "dataset.enterPathFirst": "请先输入目录路径",
      "dataset.error": "错误：{message}",
      "dataset.duplicatePaths": "错误：检测到重复的图像目录。每个数据集必须使用唯一路径。",
      "dataset.imageDirectory": "图像目录",
      "dataset.openFolder": "打开文件夹",
      "dataset.numRepeats": "重复次数",
      "dataset.keepTokens": "保留词元",
      "dataset.captionPrefix": "标题前缀",
      "dataset.captionDropout": "标题丢弃率",
      "dataset.tagDropout": "标签丢弃率",
      "dataset.dropoutEvery": "每 N 轮丢弃",
      "dataset.shuffleCaptions": "打乱标题",
      "dataset.flipAugmentations": "翻转增强",
      "dataset.cacheMetadata": "缓存元数据",
      "dataset.regularization": "正则化数据集",
      "dataset.title": "数据集",
      "dataset.disabledZero": "0 = 禁用",
      "dataset.regularizationHint": "此文件夹中的图像将作为正则化（类别图像）使用，以防止过拟合。",
      "prompts.skip": "跳过",
      "prompts.randomSeedsApplied": "已将随机种子应用到所有提示词",
      "prompts.appliedToAll": "已将全局设置应用到所有提示词",
      "prompts.baseModel": "基础模型（无 LoRA）",
      "samples.prompt": "提示词",
      "samples.showPerPrompt": "每个提示词显示",
      "samples.recent": "最近 {count} 个",
      "samples.all": "全部",
      "config.resetTitle": "重置配置",
      "config.resetMessage": "将所有设置重置为模板默认值？",
      "config.resetDone": "配置已重置为默认值",
      "settings.clearLogsTitle": "清除日志",
      "settings.clearLogsMessage": "删除此任务的所有 TensorBoard 日志？",
      "settings.logsCleared": "日志已清除",
      "confirm.cancel": "取消",
      "confirm.confirm": "确认",
      "gpu.noNvidia": "未检测到 NVIDIA GPU（仅 CPU）。",
      "gpu.noNvidiaGeneration": "未检测到 NVIDIA GPU。",
      "gpu.error": "错误：{message}",
      "progressive.enterTwo": "请先输入至少两个分辨率以配置阶段。",
      "progressive.percentOfSteps": "占总步数 {percent}%",
      "progressive.sumHint": "每个比例表示该分辨率占总步数的比例，总和必须为 1.0。",
      "progressive.sum": "总和：{sum}",
      "global.modelsSuffix": " 模型",
      "global.application": "应用程序",
      "global.settings": "⚙️ 全局设置",
      "global.path.dit_path": "DiT 模型路径",
      "global.path.qwen3_path": "Qwen3 文本编码器路径",
      "global.path.vae_path": "VAE 路径",
      "global.path.lumina_dit_path": "Lumina DiT 模型路径",
      "global.path.gemma2_path": "Gemma2 模型路径",
      "global.path.lumina_vae_path": "Lumina VAE 路径",
      "global.allInOne": "用作一体化检查点",
      "global.allInOneHint": "将第一个路径复制到此架构的其他字段。",
      "optimizer.muonWarning": "Muon 与 DeepSpeed 和 FSDP1 不兼容。多 GPU 模式已恢复为 DDP。",
      "optimizer.muonLoraWarning": "Muon 优化器不支持 LoRA 训练。已恢复为 AdamW8bit。",
    },
  };

  function normalizeLocale(value) {
    return typeof value === "string" && value.toLowerCase().startsWith("zh")
      ? "zh-CN"
      : "en";
  }

  function resolveLocale() {
    const saved = storage.getItem(STORAGE_KEY);
    return saved === "en" || saved === "zh-CN"
      ? saved
      : normalizeLocale(navigator.language);
  }

  let currentLocale = resolveLocale();

  function interpolate(value, params = {}) {
    return value.replace(/\{(\w+)\}/g, (match, key) =>
      Object.prototype.hasOwnProperty.call(params, key) ? String(params[key]) : match,
    );
  }

  function t(key, params = {}) {
    const value = translations[currentLocale][key] || translations.en[key] || key;
    return interpolate(value, params);
  }

  function translateText(value) {
    if (typeof value !== "string") return value;
    const source = value.trim();
    const reverse = Object.entries(translations["zh-CN"].text || {}).find(([, translated]) => translated === source)?.[0];
    const lookup = currentLocale === "zh-CN" ? source : (reverse || source);
    const translated = currentLocale === "zh-CN"
      ? translations["zh-CN"].text?.[lookup]
      : lookup;
    return translated ? value.replace(source, translated) : value;
  }

  function translateElement(element) {
    if (!element) return;
    if (element.dataset?.i18n) element.textContent = t(element.dataset.i18n);
    if (element.dataset?.i18nPlaceholder) {
      element.placeholder = t(element.dataset.i18nPlaceholder);
    } else if (element.placeholder) {
      element.placeholder = translateText(element.placeholder);
    }
    if (element.dataset?.i18nTitle) element.title = t(element.dataset.i18nTitle);
    else if (element.title) element.title = translateText(element.title);
    if (element.dataset?.i18nAriaLabel) {
      element.setAttribute("aria-label", t(element.dataset.i18nAriaLabel));
    }
    if (typeof element.querySelectorAll !== "function") return;
    element.querySelectorAll("[data-i18n]").forEach((child) => {
      child.textContent = t(child.dataset.i18n);
    });
    element.querySelectorAll("[data-i18n-placeholder]").forEach((child) => {
      child.placeholder = t(child.dataset.i18nPlaceholder);
    });
    element.querySelectorAll("[data-i18n-title]").forEach((child) => {
      child.title = t(child.dataset.i18nTitle);
    });
    element.querySelectorAll("[data-i18n-aria-label]").forEach((child) => {
      child.setAttribute("aria-label", t(child.dataset.i18nAriaLabel));
    });
    if (document?.createTreeWalker && typeof Node !== "undefined") {
      const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
      const textNodes = [];
      let node;
      while ((node = walker.nextNode())) textNodes.push(node);
      textNodes.forEach((textNode) => {
        if (textNode.parentElement?.closest(".job-name, .sample-name, textarea, input, code, pre, [data-user-content]")) {
          return;
        }
        textNode.nodeValue = translateText(textNode.nodeValue);
      });
    }
  }

  function updateLanguageControl() {
    if (!document || typeof document.getElementById !== "function") return;
    const button = document.getElementById("btn-language");
    if (!button) return;
    const nextKey = currentLocale === "zh-CN"
      ? "language.switchToEnglish"
      : "language.switchToChinese";
    button.textContent = t(nextKey);
    button.setAttribute("aria-label", t(nextKey));
    button.title = t(nextKey);
  }

  function translateDocument(root = document) {
    if (!root) return;
    translateElement(root);
    if (document?.title) document.title = t("document.title");
    if (typeof root.querySelectorAll === "function") {
      root.querySelectorAll("input, textarea, button, [title]").forEach((element) => {
        if (!element.dataset?.i18nPlaceholder && element.placeholder) {
          element.placeholder = translateText(element.placeholder);
        }
        if (!element.dataset?.i18nTitle && element.title) {
          element.title = translateText(element.title);
        }
      });
    }
    updateLanguageControl();
  }

  function setLocale(locale) {
    currentLocale = locale === "zh-CN" ? "zh-CN" : "en";
    storage.setItem(STORAGE_KEY, currentLocale);
    if (document?.documentElement) document.documentElement.lang = currentLocale;
    translateDocument();
    if (document?.dispatchEvent) {
      const event = CustomEvent
        ? new CustomEvent("localechange", { detail: { locale: currentLocale } })
        : { type: "localechange", detail: { locale: currentLocale } };
      document.dispatchEvent(event);
    }
  }

  function toggleLocale() {
    setLocale(currentLocale === "zh-CN" ? "en" : "zh-CN");
  }

  if (document?.documentElement) document.documentElement.lang = currentLocale;

  return {
    getLocale: () => currentLocale,
    setLocale,
    toggleLocale,
    t,
    translateText,
    translateElement,
    translateDocument,
    getPathLabel: (key, fallback) => {
      const translated = t(key);
      return translated === key ? fallback : translated;
    },
  };
});
