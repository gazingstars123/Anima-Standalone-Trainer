const express = require('express');
const { spawn, exec } = require('child_process');
const fs = require('fs');
const path = require('path');
const TOML = require('@iarna/toml');
const net = require('net');
const http = require('http');
const WebSocket = require('ws');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

// Parse CLI argument for port
const args = process.argv.slice(2);
const portArg = args.find(a => a.startsWith('--port='));
const DEFAULT_PORT = portArg
    ? parseInt(portArg.split('=')[1])
    : (parseInt(args[0]) || 3000);

// Paths
const ROOT_DIR = path.join(__dirname, '..');
const JOBS_DIR = path.join(__dirname, 'jobs');
const TEMPLATES_DIR = path.join(__dirname, 'templates');
const GLOBAL_CONFIG_PATH = path.join(__dirname, 'global_config.toml');
const TRAINING_SCRIPT = path.join(ROOT_DIR, 'anima_train_network.py');
const UPLOAD_DIR = path.join(__dirname, 'public', 'uploads');

// Middleware
app.use(express.json({ limit: '50mb' }));
app.use(express.static(path.join(__dirname, 'public')));

// Ensure directories exist
if (!fs.existsSync(JOBS_DIR)) {
    fs.mkdirSync(JOBS_DIR, { recursive: true });
}
if (!fs.existsSync(UPLOAD_DIR)) {
    fs.mkdirSync(UPLOAD_DIR, { recursive: true });
}

// Ensure global config exists from template
const TEMPLATE_CONFIG_PATH = path.join(__dirname, 'global_config.template.toml');
if (!fs.existsSync(GLOBAL_CONFIG_PATH) && fs.existsSync(TEMPLATE_CONFIG_PATH)) {
    try {
        fs.copyFileSync(TEMPLATE_CONFIG_PATH, GLOBAL_CONFIG_PATH);
        console.log("Created global_config.toml from template.");
    } catch (e) {
        console.error("Failed to create global_config.toml from template:", e);
    }
}

// Track running processes
const runningJobs = new Map();

// WebSocket clients per job
const wsClients = new Map(); // jobName -> Set<ws>

// --- Helper Functions ---

function sanitizeName(name) {
    let safe = name.replace(/[<>:"/\\|?*]/g, '').trim();
    if (!safe) safe = 'job_' + Date.now();
    return safe;
}

function getJobPath(name) {
    return path.join(JOBS_DIR, sanitizeName(name));
}

function getGlobalConfig() {
    if (fs.existsSync(GLOBAL_CONFIG_PATH)) {
        try {
            const config = TOML.parse(fs.readFileSync(GLOBAL_CONFIG_PATH, 'utf8'));
            // Sanitize paths (remove extra quotes if user added them manually)
            if (config.model_paths) {
                for (const key in config.model_paths) {
                    if (typeof config.model_paths[key] === 'string') {
                        config.model_paths[key] = config.model_paths[key].replace(/^"|"$/g, '').replace(/^'|'$/g, '');
                    }
                }
            }
            return config;
        } catch (err) {
            console.error('Failed to parse global config:', err.message);
        }
    }
    return {
        model_paths: {
            dit_path: '',
            qwen3_path: '',
            vae_path: ''
        },
        venv_path: path.join(ROOT_DIR, 'venv')
    };
}

app.get('/api/gpu/activity', (req, res) => {
    const activity = {};

    // Check running jobs (training/generation)
    for (const [name, job] of runningJobs.entries()) {
        if (job.gpuIds) {
            job.gpuIds.split(',').forEach(id => {
                const trimmed = id.trim();
                if (trimmed) {
                    activity[trimmed] = job.type === 'generation' ? 'sampling' : 'training';
                }
            });
        }
    }

    // Check persistent generation
    if (persistentGenProcess && persistentGenProcess.gpuIds) {
        persistentGenProcess.gpuIds.split(',').forEach(id => {
            const trimmed = id.trim();
            if (trimmed) activity[trimmed] = 'sampling';
        });
    }

    res.json(activity);
});

// Get GPU Information using nvidia-smi with Python fallback
async function getDetectedGPUs() {
    return new Promise((resolve) => {
        // 1. Try nvidia-smi
        const smi = spawn('nvidia-smi', ['--query-gpu=index,name,memory.total', '--format=csv,noheader']);
        let stdout = '';
        let stderr = '';

        smi.stdout.on('data', (data) => stdout += data);
        smi.stderr.on('data', (data) => stderr += data);

        smi.on('close', (code) => {
            if (code === 0 && stdout) {
                const gpus = stdout.trim().split('\n').map(line => {
                    const parts = line.split(',').map(s => s.trim());
                    if (parts.length < 3) return null;
                    return {
                        index: parseInt(parts[0]),
                        name: parts[1],
                        memory: parts[2]
                    };
                }).filter(g => g !== null);
                return resolve(gpus);
            }

            // 2. Fallback to Python (torch)
            console.warn("nvidia-smi failed, trying python fallback...");
            const globalConfig = getGlobalConfig();
            const venvPath = globalConfig.venv_path || path.join(ROOT_DIR, 'venv');
            let pythonPath = 'python'; // Default
            if (process.platform === 'win32') {
                pythonPath = path.join(venvPath, 'Scripts', 'python.exe');
            } else {
                pythonPath = path.join(venvPath, 'bin', 'python');
            }

            if (!fs.existsSync(pythonPath)) {
                pythonPath = 'python';
            }

            const pyScript = "import torch; import json; print(json.dumps([{'index': i, 'name': torch.cuda.get_device_name(i), 'memory': f'{torch.cuda.get_device_properties(i).total_memory // 1024**2} MiB'} for i in range(torch.cuda.device_count())]))";

            const pyProc = spawn(pythonPath, ['-c', pyScript]);
            let pyOut = '';
            let pyErr = '';

            pyProc.stdout.on('data', (data) => pyOut += data);
            pyProc.stderr.on('data', (data) => pyErr += data);

            pyProc.on('close', (pyCode) => {
                if (pyCode !== 0) {
                    console.error("Python GPU detection failed:", pyErr);
                    return resolve([]);
                }
                try {
                    const gpus = JSON.parse(pyOut.trim());
                    resolve(gpus);
                } catch (e) {
                    console.error("Failed to parse Python GPU output:", e);
                    resolve([]);
                }
            });
        });

        smi.on('error', (err) => {
            // Silently fail to fallback
        });
    });
}

function getDefaultConfig() {
    const templatePath = path.join(TEMPLATES_DIR, 'config_template.toml');
    if (fs.existsSync(templatePath)) {
        try {
            return { config: TOML.parse(fs.readFileSync(templatePath, 'utf8')), useFallback: false };
        } catch (err) {
            console.error('Config template parse error:', err.message);
        }
    }
    return {
        config: {
            training_arguments: {
                output_name: 'my_anima_lora',
                learning_rate: 5e-5,
                max_train_epochs: 20,
                mixed_precision: 'bf16'
            },
            network_arguments: {
                network_module: 'networks.lora_anima',
                network_dim: 16,
                network_alpha: 16
            }
        },
        useFallback: true
    };
}

function getDefaultDataset() {
    const templatePath = path.join(TEMPLATES_DIR, 'dataset_template.toml');
    if (fs.existsSync(templatePath)) {
        try {
            return TOML.parse(fs.readFileSync(templatePath, 'utf8'));
        } catch (err) {
            console.error('Dataset template parse error:', err.message);
        }
    }
    return {
        general: { enable_bucket: true },
        datasets: [{ resolution: [1536, 1536], batch_size: 4, caption_extension: '.txt', subsets: [{ image_dir: '', num_repeats: 1 }] }]
    };
}

// Startup validation
(function validateTemplates() {
    const configTemplate = path.join(TEMPLATES_DIR, 'config_template.toml');
    const datasetTemplate = path.join(TEMPLATES_DIR, 'dataset_template.toml');
    [configTemplate, datasetTemplate].forEach(f => {
        if (fs.existsSync(f)) {
            try {
                TOML.parse(fs.readFileSync(f, 'utf8'));
                console.log(`✅ Template validated: ${path.basename(f)}`);
            } catch (err) {
                console.error(`❌ Template error in ${path.basename(f)}: ${err.message}`);
            }
        } else {
            console.warn(`⚠️  Template not found: ${path.basename(f)}`);
        }
    });
})();

function broadcastLog(jobName, message) {
    const clients = wsClients.get(jobName);
    if (clients) {
        const data = JSON.stringify({ job: jobName, type: 'log', data: message });
        clients.forEach(ws => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(data);
            }
        });
    }
}

function getTrainingFailureHints(exitCode, logBuffer) {
    if (exitCode === 0) return [];

    const joinedLogs = logBuffer.join('');
    const hints = [];

    // On Windows, accelerate can exit with code 1 while reporting a child process crash
    // with NTSTATUS 3221225477 (0xC0000005, access violation).
    if (joinedLogs.includes('3221225477') || joinedLogs.includes('0xC0000005')) {
        hints.push(
            'Detected Windows access violation (3221225477 / 0xC0000005) from a child training process.',
            'Common causes: unstable overclock/driver state, aggressive worker/pinned-memory settings, CUDA/xFormers mismatch, or mixed-precision kernel instability.',
            'Suggested checks: update/reinstall GPU driver, reduce dataloader workers, disable xFormers/FlashAttention options, and retry with safer precision settings.'
        );
    }

    return hints;
}

function broadcastStatus(jobName, status) {
    const clients = wsClients.get(jobName);
    if (clients) {
        const data = JSON.stringify({ job: jobName, type: 'status', data: status });
        clients.forEach(ws => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(data);
            }
        });
    }
}

// Build the full TOML config file for training, merging global model paths + job paths
function buildTrainingConfig(jobName, jobPath) {
    const globalConfig = getGlobalConfig();
    const configPath = path.join(jobPath, 'config.toml');
    const jobConfig = TOML.parse(fs.readFileSync(configPath, 'utf8'));

    const outputDir = path.join(jobPath, 'output');
    const loggingDir = path.join(jobPath, 'logs');
    const datasetConfigPath = path.join(jobPath, 'dataset.toml');
    const samplePromptsPath = path.join(jobPath, 'sample_prompts.txt');

    // Build the merged config
    const merged = {};

    // Model arguments from global config
    merged.model_arguments = {
        dit_path: globalConfig.model_paths?.dit_path || '',
        qwen3_path: globalConfig.model_paths?.qwen3_path || '',
        vae_path: globalConfig.model_paths?.vae_path || ''
    };

    // Dataset arguments
    merged.dataset_arguments = {
        dataset_config: datasetConfigPath,
        cache_latents_to_disk: jobConfig.training_arguments?.cache_latents_to_disk ?? true,
        cache_text_encoder_outputs_to_disk: jobConfig.training_arguments?.cache_text_encoder_outputs_to_disk ?? true
    };

    // Training arguments (remove cache args since they're in dataset_arguments)
    const trainingArgs = { ...jobConfig.training_arguments };
    delete trainingArgs.cache_latents_to_disk;
    delete trainingArgs.cache_text_encoder_outputs_to_disk;
    merged.training_arguments = {
        ...trainingArgs,
        output_dir: outputDir,
        logging_dir: loggingDir,
        save_state: true,
        save_last_n_steps_state: 1,
        save_last_n_epochs_state: 1
    };

    // Move resume from network_args to training_args
    if (jobConfig.network_arguments?.resume) {
        merged.training_arguments.resume = jobConfig.network_arguments.resume.replace(/^['"]+|['"]+$/g, '');
    }

    // Add sample prompts if file exists and has content
    if (fs.existsSync(samplePromptsPath)) {
        const prompts = fs.readFileSync(samplePromptsPath, 'utf8').trim();
        if (prompts.length > 0) {
            const ta = jobConfig.training_arguments || {};
            if (ta.sample_every_n_steps || ta.sample_every_n_epochs) {
                merged.sample_arguments = {
                    sample_prompts: samplePromptsPath
                };

                //prefer steps if set, otherwise epochs
                if (ta.sample_every_n_steps) {
                    merged.sample_arguments.sample_every_n_steps = ta.sample_every_n_steps;
                } else {
                    merged.sample_arguments.sample_every_n_epochs = ta.sample_every_n_epochs || 1;
                }
            }
        }
    }

    delete merged.training_arguments.sample_every_n_epochs;
    delete merged.training_arguments.sample_every_n_steps;

    // Network arguments
    merged.network_arguments = { ...jobConfig.network_arguments };
    if (merged.network_arguments.network_weights) {
        merged.network_arguments.network_weights = merged.network_arguments.network_weights.replace(/^['"]+|['"]+$/g, '');
    }
    delete merged.network_arguments.resume;

    // Anima arguments
    if (jobConfig.anima_arguments) {
        merged.anima_arguments = { ...jobConfig.anima_arguments };
    }

    return merged;
}

// --- WebSocket ---

wss.on('connection', (ws) => {
    ws.subscribedJob = null;

    ws.on('message', (message) => {
        try {
            const msg = JSON.parse(message);
            if (msg.type === 'subscribe' && msg.job) {
                // Unsubscribe from previous
                if (ws.subscribedJob) {
                    const oldClients = wsClients.get(ws.subscribedJob);
                    if (oldClients) oldClients.delete(ws);
                }
                // Subscribe to new
                ws.subscribedJob = msg.job;
                if (!wsClients.has(msg.job)) {
                    wsClients.set(msg.job, new Set());
                }
                wsClients.get(msg.job).add(ws);

                // Send current status
                const isRunning = runningJobs.has(msg.job);
                ws.send(JSON.stringify({
                    job: msg.job,
                    type: 'status',
                    data: isRunning ? 'running' : 'idle'
                }));

                // Send buffered logs
                const jobData = runningJobs.get(msg.job);
                if (jobData && jobData.logBuffer) {
                    ws.send(JSON.stringify({
                        job: msg.job,
                        type: 'log',
                        data: jobData.logBuffer.join('')
                    }));
                }
            }
        } catch (e) {
            // ignore
        }
    });

    ws.on('close', () => {
        if (ws.subscribedJob) {
            const clients = wsClients.get(ws.subscribedJob);
            if (clients) clients.delete(ws);
        }
    });
});

// --- Global Config API ---

app.get('/api/global-config', (req, res) => {
    try {
        res.json(getGlobalConfig());
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.put('/api/global-config', (req, res) => {
    try {
        const tomlStr = TOML.stringify(req.body);
        fs.writeFileSync(GLOBAL_CONFIG_PATH, tomlStr, 'utf8');
        res.json({ success: true });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// --- Background Image API ---

app.post('/api/global/background', (req, res) => {
    try {
        const { image } = req.body;
        if (!image) return res.status(400).json({ error: 'No image data' });

        const base64Data = image.replace(/^data:image\/\w+;base64,/, '');
        const extension = image.split(';')[0].split('/')[1];
        const filename = `bg_${Date.now()}.${extension}`;
        const filePath = path.join(UPLOAD_DIR, filename);

        // Delete old backgrounds
        if (fs.existsSync(UPLOAD_DIR)) {
            fs.readdirSync(UPLOAD_DIR).forEach(file => fs.unlinkSync(path.join(UPLOAD_DIR, file)));
        }

        fs.writeFileSync(filePath, base64Data, 'base64');
        res.json({ success: true, url: `/uploads/${filename}` });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.delete('/api/global/background', (req, res) => {
    try {
        if (fs.existsSync(UPLOAD_DIR)) {
            fs.readdirSync(UPLOAD_DIR).forEach(file => fs.unlinkSync(path.join(UPLOAD_DIR, file)));
        }
        res.json({ success: true });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// --- System API Routes ---

app.get('/api/system/gpus', async (req, res) => {
    const gpus = await getDetectedGPUs();
    res.json(gpus);
});

// --- Job API Routes ---

// List all jobs
app.get('/api/jobs', (req, res) => {
    try {
        if (!fs.existsSync(JOBS_DIR)) return res.json([]);
        const jobs = fs.readdirSync(JOBS_DIR, { withFileTypes: true })
            .filter(d => d.isDirectory())
            .map(d => {
                const configPath = path.join(JOBS_DIR, d.name, 'config.toml');
                const hasConfig = fs.existsSync(configPath);
                let mtime = 0;
                if (hasConfig) {
                    try { mtime = fs.statSync(configPath).mtimeMs; } catch (e) { }
                }
                return {
                    name: d.name,
                    hasConfig,
                    running: runningJobs.has(d.name),
                    mtime
                };
            })
            .sort((a, b) => b.mtime - a.mtime);
        res.json(jobs);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Create new job
app.post('/api/jobs', (req, res) => {
    try {
        const { name } = req.body;
        if (!name) return res.status(400).json({ error: 'Name required' });

        const safeName = sanitizeName(name);
        const jobPath = path.join(JOBS_DIR, safeName);

        if (fs.existsSync(jobPath)) {
            return res.status(409).json({ error: 'Job already exists' });
        }

        // Create directory structure
        fs.mkdirSync(jobPath, { recursive: true });
        fs.mkdirSync(path.join(jobPath, 'output'), { recursive: true });
        fs.mkdirSync(path.join(jobPath, 'logs'), { recursive: true });
        fs.mkdirSync(path.join(jobPath, 'samples'), { recursive: true });

        // Copy template configs
        const { config, useFallback } = getDefaultConfig();
        fs.writeFileSync(path.join(jobPath, 'config.toml'), TOML.stringify(config), 'utf8');

        const datasetConfig = getDefaultDataset();
        fs.writeFileSync(path.join(jobPath, 'dataset.toml'), TOML.stringify(datasetConfig), 'utf8');

        // Copy sample prompts template
        const promptsTemplate = path.join(TEMPLATES_DIR, 'sample_prompts.txt');
        if (fs.existsSync(promptsTemplate)) {
            fs.copyFileSync(promptsTemplate, path.join(jobPath, 'sample_prompts.txt'));
        } else {
            fs.writeFileSync(path.join(jobPath, 'sample_prompts.txt'), '', 'utf8');
        }

        res.json({ name: safeName, path: jobPath, useFallback });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Get job config
app.get('/api/jobs/:name', (req, res) => {
    try {
        const jobPath = getJobPath(req.params.name);
        const configPath = path.join(jobPath, 'config.toml');
        const datasetPath = path.join(jobPath, 'dataset.toml');

        if (!fs.existsSync(configPath)) {
            return res.status(404).json({ error: 'Job not found' });
        }

        const config = TOML.parse(fs.readFileSync(configPath, 'utf8'));
        const dataset = fs.existsSync(datasetPath)
            ? TOML.parse(fs.readFileSync(datasetPath, 'utf8'))
            : getDefaultDataset();

        res.json({ name: req.params.name, config, dataset });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Update job config
app.put('/api/jobs/:name', (req, res) => {
    try {
        const jobPath = getJobPath(req.params.name);
        if (!fs.existsSync(jobPath)) {
            return res.status(404).json({ error: 'Job not found' });
        }

        if (req.body.config) {
            fs.writeFileSync(path.join(jobPath, 'config.toml'), TOML.stringify(req.body.config), 'utf8');
        }
        if (req.body.dataset) {
            fs.writeFileSync(path.join(jobPath, 'dataset.toml'), TOML.stringify(req.body.dataset), 'utf8');
        }

        res.json({ success: true });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Delete job
app.delete('/api/jobs/:name', (req, res) => {
    try {
        const jobPath = getJobPath(req.params.name);
        if (runningJobs.has(req.params.name)) {
            return res.status(400).json({ error: 'Stop job before deleting' });
        }
        if (fs.existsSync(jobPath)) {
            fs.rmSync(jobPath, { recursive: true });
        }
        res.json({ success: true });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Clone job
app.post('/api/jobs/:name/clone', (req, res) => {
    try {
        const sourceName = sanitizeName(req.params.name);
        const sourcePath = getJobPath(sourceName);

        if (!fs.existsSync(sourcePath)) {
            return res.status(404).json({ error: 'Source job not found' });
        }

        let targetName = req.body.newName ? sanitizeName(req.body.newName) : null;

        // Auto-generate name if not provided
        if (!targetName) {
            targetName = `${sourceName}_copy`;
            let counter = 1;
            while (fs.existsSync(getJobPath(targetName))) {
                counter++;
                targetName = `${sourceName}_copy_${counter}`;
            }
        }

        const targetPath = getJobPath(targetName);
        if (fs.existsSync(targetPath)) {
            return res.status(409).json({ error: `Job "${targetName}" already exists` });
        }

        fs.mkdirSync(targetPath, { recursive: true });
        fs.mkdirSync(path.join(targetPath, 'output'), { recursive: true });
        fs.mkdirSync(path.join(targetPath, 'logs'), { recursive: true });
        fs.mkdirSync(path.join(targetPath, 'samples'), { recursive: true });

        // Copy config files
        ['dataset.toml', 'sample_prompts.txt'].forEach(file => {
            const src = path.join(sourcePath, file);
            if (fs.existsSync(src)) {
                fs.copyFileSync(src, path.join(targetPath, file));
            }
        });

        // Handle config.toml special case: update output_name
        const configSrc = path.join(sourcePath, 'config.toml');
        if (fs.existsSync(configSrc)) {
            let config = TOML.parse(fs.readFileSync(configSrc, 'utf8'));

            // Sync output_name with new job name
            if (!config.training_arguments) config.training_arguments = {};
            config.training_arguments.output_name = targetName;

            fs.writeFileSync(path.join(targetPath, 'config.toml'), TOML.stringify(config), 'utf8');
        }

        res.json({ success: true, name: targetName });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// --- Prompts API ---

app.get('/api/jobs/:name/prompts', (req, res) => {
    try {
        const jobPath = getJobPath(req.params.name);
        const promptsPath = path.join(jobPath, 'sample_prompts.txt');
        if (!fs.existsSync(promptsPath)) {
            return res.json({ prompts: [] });
        }
        const text = fs.readFileSync(promptsPath, 'utf8').trim();
        const prompts = text ? text.split('\n').map(l => l.trim()).filter(l => l.length > 0) : [];
        res.json({ prompts });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.put('/api/jobs/:name/prompts', (req, res) => {
    try {
        const jobPath = getJobPath(req.params.name);
        const promptsPath = path.join(jobPath, 'sample_prompts.txt');
        const prompts = req.body.prompts || [];
        fs.writeFileSync(promptsPath, prompts.join('\n') + (prompts.length ? '\n' : ''), 'utf8');
        res.json({ success: true });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

let persistentGenProcess = null; // { process, port, jobName }
const GEN_SERVER_PORT = 5000; // Fixed port for now, or dynamic

// Cross-platform process killer
function killProcess(pid) {
    return new Promise((resolve) => {
        try {
            if (process.platform === 'win32') {
                const k = spawn('taskkill', ['/PID', pid.toString(), '/F', '/T']);
                k.on('close', () => resolve());
                k.on('error', () => resolve());
            } else {
                // Linux/Mac: kill -9
                try {
                    process.kill(pid, 'SIGKILL');
                    resolve();
                } catch (e) {
                    resolve();
                }
            }
        } catch (e) {
            resolve();
        }
    });
}

// --- Cross-platform venv/spawn helpers ---

const isWindows = process.platform === 'win32';

function getVenvPaths(venvPath) {
    if (isWindows) {
        return {
            activate: path.join(venvPath, 'Scripts', 'Activate.ps1'),
            accelerate: path.join(venvPath, 'Scripts', 'accelerate.exe'),
            python: path.join(venvPath, 'Scripts', 'python.exe'),
        };
    } else {
        return {
            activate: path.join(venvPath, 'bin', 'activate'),
            accelerate: path.join(venvPath, 'bin', 'accelerate'),
            python: path.join(venvPath, 'bin', 'python'),
        };
    }
}

function buildEnvVar(name, value) {
    return isWindows ? `$env:${name}='${value}';` : `export ${name}='${value}';`;
}

function buildShellScript(activatePath, envVars, command) {
    if (isWindows) {
        return `& "${activatePath}";\n${envVars}\n${command}`;
    } else {
        return `source "${activatePath}"\n${envVars}\n${command}`;
    }
}

function spawnShell(script, cwd) {
    if (isWindows) {
        return spawn('powershell', ['-NoProfile', '-Command', script], {
            cwd,
            stdio: ['pipe', 'pipe', 'pipe']
        });
    } else {
        return spawn('bash', ['-c', script], {
            cwd,
            stdio: ['pipe', 'pipe', 'pipe']
        });
    }
}

function killPersistentGen() {
    if (persistentGenProcess) {
        console.log(`Stop persistent gen server (PID: ${persistentGenProcess.process.pid})`);
        try {
            // Try graceful stop via API first
            fetch(`http://localhost:${persistentGenProcess.port}/stop`, { method: 'POST' }).catch(() => { });

            // Force kill after short delay
            setTimeout(() => {
                if (persistentGenProcess && persistentGenProcess.process) {
                    killProcess(persistentGenProcess.process.pid);
                    persistentGenProcess = null;
                }
            }, 1000);
        } catch (e) {
            persistentGenProcess = null;
        }
    }
}

app.post('/api/jobs/:name/unload', (req, res) => {
    try {
        if (persistentGenProcess) {
            killPersistentGen();
            res.json({ success: true, message: "Model unloaded" });
        } else {
            res.json({ success: true, message: "No model loaded" });
        }
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.post('/api/jobs/:name/train/stop', async (req, res) => {
    try {
        const jobName = sanitizeName(req.params.name);
        const job = runningJobs.get(jobName);

        if (!job) {
            return res.status(400).json({ error: 'Job not running' });
        }

        // Kill the process tree
        if (job.pid) {
            await killProcess(job.pid);
        }

        runningJobs.delete(jobName);
        broadcastStatus(jobName, 'idle');
        res.json({ success: true });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.post('/api/jobs/:name/tensorboard/stop', async (req, res) => {
    try {
        const jobName = sanitizeName(req.params.name);
        const tb = tbProcesses.get(jobName);

        if (!tb) {
            return res.json({ success: true, message: 'Not running' });
        }

        if (tb.pid) {
            await killProcess(tb.pid);
        }

        tbProcesses.delete(jobName);
        res.json({ success: true });

    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});


app.get('/api/jobs/:name/checkpoints', (req, res) => {
    try {
        const jobPath = getJobPath(req.params.name);
        const outputDir = path.join(jobPath, 'output');
        if (!fs.existsSync(outputDir)) return res.json([]);

        const files = fs.readdirSync(outputDir)
            .filter(f => f.endsWith('.safetensors'))
            .map(f => {
                const stat = fs.statSync(path.join(outputDir, f));
                return { name: f, path: path.join(outputDir, f), mtime: stat.mtimeMs };
            })
            .sort((a, b) => b.mtime - a.mtime);

        res.json(files);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.post('/api/jobs/:name/generate', async (req, res) => {
    try {
        const jobName = sanitizeName(req.params.name);
        if (runningJobs.has(jobName)) {
            return res.status(400).json({ error: 'Job is running. Stop it first.' });
        }

        const jobPath = getJobPath(jobName);
        const configPath = path.join(jobPath, 'config.toml');

        if (!fs.existsSync(configPath)) {
            return res.status(404).json({ error: 'Job not found' });
        }

        // Merged config for paths/args
        const mergedConfig = buildTrainingConfig(jobName, jobPath);

        const outputDir = path.join(jobPath, 'output');
        const promptsPath = path.join(jobPath, 'sample_prompts.txt');

        if (!fs.existsSync(promptsPath) || fs.readFileSync(promptsPath, 'utf8').trim().length === 0) {
            return res.status(400).json({ error: 'No sample prompts found. Add prompts in the Prompts tab.' });
        }

        const globalConfig = getGlobalConfig();
        const venvPath = globalConfig.venv_path || path.join(ROOT_DIR, 'venv');
        const venv = getVenvPaths(venvPath);
        const genScript = path.join(ROOT_DIR, 'anima_gen.py');

        // Extract args
        const mArgs = mergedConfig.model_arguments;
        const tArgs = mergedConfig.training_arguments;
        const aArgs = mergedConfig.anima_arguments || {};

        // Read config to check for GPU IDs - prefer gen-specific GPU selection
        let gpuEnv = '';
        const genGpuIds = req.body.gen_gpu_ids || '';
        const rawConfig = TOML.parse(fs.readFileSync(configPath, 'utf8'));
        const configGpuIds = rawConfig.gpu_ids ? rawConfig.gpu_ids.toString().trim() : '';

        const currentGpuIdsRaw = genGpuIds || configGpuIds;
        const genGpuIdsNormalized = currentGpuIdsRaw.split(',').map(s => s.trim()).filter(s => s.length > 0).sort().join(',');

        if (genGpuIdsNormalized) {
            if (/^[\d\s,]+$/.test(genGpuIdsNormalized)) {
                gpuEnv = buildEnvVar('CUDA_VISIBLE_DEVICES', genGpuIdsNormalized);
                console.log(`[Gen] Using GPU isolation: ${gpuEnv}`);
            }
        }

        const args = [
            `--dit_path="${mArgs.dit_path}"`,
            `--qwen3_path="${mArgs.qwen3_path}"`,
            `--vae_path="${mArgs.vae_path}"`,
            `--sample_prompts="${promptsPath}"`,
            `--output_dir="${outputDir}"`,
            `--output_name="${tArgs.output_name || 'baseline'}"`,
            `--mixed_precision="${tArgs.mixed_precision || 'bf16'}"`,
            `--seed=${tArgs.seed || 42}`,
            `--discrete_flow_shift=${aArgs.discrete_flow_shift || 3.0}`,
            `--timestep_sample_method="${aArgs.timestep_sample_method || 'logit_normal'}"`
        ];

        // Attention support
        if (req.body.flash_attn) {
            args.push('--flash_attn');
        } else if (req.body.sage_attn) {
            args.push('--sage_attn');
        }

        // Multi-GPU model sharding support
        const genGpuCount = genGpuIdsNormalized ? genGpuIdsNormalized.split(',').length : 0;
        let genAccelerateFlags = '';
        if (genGpuCount > 1) {
            const multiGpuMode = req.body.gen_multi_gpu_mode || 'parallel_cfg';
            args.push(`--device_map=${multiGpuMode}`);
            // Force single process — both modes run one process across all GPUs
            genAccelerateFlags = '--num_processes 1';
        }

        // LoRA support
        if (req.body.network_weights) {
            const nw = req.body.network_weights.replace(/^['"]+|['"]+$/g, '');
            args.push(`--network_weights="${nw}"`);
            args.push(`--network_mul=${req.body.network_mul || 1.0}`);
        }

        const keepLoaded = req.body.keep_loaded === true;

        // Ensure logs dir exists
        const logsDir = path.join(jobPath, 'logs');
        if (!fs.existsSync(logsDir)) fs.mkdirSync(logsDir, { recursive: true });

        // Logic for Persistent vs One-Shot
        if (keepLoaded) {
            // Check if we need to start the server
            // If running but different job (unlikely since we enforce one job), kill it?
            // Actually Anima is single model so if job changes we might need to reload. 
            // For simplicity, if current persistent process jobName != requested jobName, restart.

            const multiGpuMode = req.body.gen_multi_gpu_mode || 'parallel_cfg';
            const flashAttn = req.body.flash_attn || false;

            if (persistentGenProcess && (
                persistentGenProcess.jobName !== jobName ||
                persistentGenProcess.gpuIds !== genGpuIdsNormalized ||
                persistentGenProcess.multiGpuMode !== multiGpuMode
            )) {
                console.log("Configuration changed, restarting persistent server...");
                killPersistentGen();
            }

            if (!persistentGenProcess) {
                const port = await findAvailablePort(GEN_SERVER_PORT);
                args.push(`--server_port=${port}`);

                const envVars = [
                    buildEnvVar('PYTHONIOENCODING', 'utf-8'),
                    buildEnvVar('LOG_LEVEL', 'DEBUG'),
                    gpuEnv
                ].filter(Boolean).join('\n');
                const launchCmd = `python -m accelerate.commands.launch --num_cpu_threads_per_process 1 ${genAccelerateFlags} "${genScript}" ${args.join(' ')}`;
                const script = buildShellScript(venv.activate, envVars, launchCmd);

                console.log("Starting persistent generation server...");
                const proc = spawnShell(script, ROOT_DIR);

                persistentGenProcess = {
                    process: proc, port, jobName,
                    gpuIds: genGpuIdsNormalized,
                    multiGpuMode: multiGpuMode,
                    flashAttn: req.body.flash_attn || false,
                    sageAttn: req.body.sage_attn || false
                };

                // Stream output
                const logFileName = `gen_server_${new Date().toISOString().replace(/[:.]/g, '-')}.log`;
                const logStream = fs.createWriteStream(path.join(logsDir, logFileName), { flags: 'a' });

                const appendLog = (data) => {
                    const text = data.toString();
                    logStream.write(text);
                    broadcastLog(jobName, text);
                };
                proc.stdout.on('data', appendLog);
                proc.stderr.on('data', appendLog);

                proc.on('close', (code) => {
                    console.log(`Persistent server exited with code ${code}`);
                    if (persistentGenProcess && persistentGenProcess.process === proc) {
                        persistentGenProcess = null;
                    }
                });

                // Wait for server to be ready (ping loop)
                let attempts = 0;
                while (attempts < 60) { // 60s timeout
                    await new Promise(r => setTimeout(r, 1000));
                    try {
                        const ping = await fetch(`http://localhost:${port}/ping`);
                        if (ping.ok) break;
                    } catch (e) { }
                    attempts++;
                }
                if (attempts >= 60) {
                    killPersistentGen();
                    return res.status(500).json({ error: "Failed to start persistent generation server (timeout)" });
                }
            }

            // Send generation request
            const payload = {
                sample_prompts: promptsPath,
                network_weights: req.body.network_weights ? req.body.network_weights.replace(/^['"]+|['"]+$/g, '') : null,
                network_mul: req.body.network_mul || 1.0,
                flash_attn: req.body.flash_attn || false,
                sage_attn: req.body.sage_attn || false
            };

            const response = await fetch(`http://localhost:${persistentGenProcess.port}/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const result = await response.json();

            if (!result.success) throw new Error(result.error);

            res.json({ success: true, message: "Generation completed (Server kept running)" });
            return;

        } else {
            // One-shot mode requested
            // If persistent server is running, kill it first to ensure clean state? 
            // Or just leave it? User asked to NOT keep loaded. 
            // Interpretation: "I want to run this once and clear memory."
            // So we should kill any persistent process.
            if (persistentGenProcess) {
                killPersistentGen();
            }

            // Standard One-Shot Logic
            const oneShotEnvVars = [
                buildEnvVar('PYTHONIOENCODING', 'utf-8'),
                gpuEnv
            ].filter(Boolean).join('\n');
            const oneShotCmd = `python -m accelerate.commands.launch --num_cpu_threads_per_process 1 ${genAccelerateFlags} "${genScript}" ${args.join(' ')}`;
            const oneShotScript = buildShellScript(venv.activate, oneShotEnvVars, oneShotCmd);

            const oneShotProc = spawnShell(oneShotScript, ROOT_DIR);

            // Write logs to file
            const oneShotLogFileName = `gen_${new Date().toISOString().replace(/[:.]/g, '-')}.log`;
            const oneShotLogStream = fs.createWriteStream(path.join(logsDir, oneShotLogFileName), { flags: 'a' });

            const oneShotAppendLog = (data) => {
                const text = data.toString();
                oneShotLogStream.write(text);
                broadcastLog(jobName, text);
            };

            oneShotProc.stdout.on('data', oneShotAppendLog);
            oneShotProc.stderr.on('data', oneShotAppendLog);

            // Prevent crashes on stream errors
            oneShotProc.stdout.on('error', (err) => console.error(`[Gen/stdout] ${err.message}`));
            oneShotProc.stderr.on('error', (err) => console.error(`[Gen/stderr] ${err.message}`));
            oneShotLogStream.on('error', (err) => console.error(`[Gen/LogFile] ${err.message}`));

            oneShotProc.on('close', (code) => {
                const msg = `\n--- Generation finished (exit code: ${code}) ---\n`;
                oneShotLogStream.write(msg);
                oneShotLogStream.end();
                broadcastLog(jobName, msg);
                runningJobs.delete(jobName);
                broadcastStatus(jobName, 'idle');
            });

            runningJobs.set(jobName, {
                process: oneShotProc,
                pid: oneShotProc.pid,
                startTime: Date.now(),
                type: 'generation',
                gpuIds: currentGpuIds
            });

            broadcastStatus(jobName, 'generating');
            res.json({ success: true, pid: oneShotProc.pid });


        }
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// --- Training Control ---

app.post('/api/jobs/:name/train/start', async (req, res) => {
    try {
        const jobName = sanitizeName(req.params.name);
        const jobPath = getJobPath(jobName);
        const configPath = path.join(jobPath, 'config.toml');

        if (!fs.existsSync(configPath)) {
            return res.status(404).json({ error: 'Job not found' });
        }
        if (runningJobs.has(jobName)) {
            return res.status(400).json({ error: 'Job already running' });
        }

        // Auto-kill persistent gen server to free VRAM
        if (persistentGenProcess) {
            console.log("Stopping persistent generation server before training...");
            killPersistentGen();
        }

        // Build merged config and write to temp file
        const mergedConfig = buildTrainingConfig(jobName, jobPath);
        const mergedConfigPath = path.join(jobPath, '_merged_config.toml');
        fs.writeFileSync(mergedConfigPath, TOML.stringify(mergedConfig), 'utf8');

        // Ensure output dirs exist
        const outputDir = path.join(jobPath, 'output');
        const logsDir = path.join(jobPath, 'logs');
        if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });
        if (!fs.existsSync(logsDir)) fs.mkdirSync(logsDir, { recursive: true });

        // Get venv path from global config
        const globalConfig = getGlobalConfig();
        const venvPath = globalConfig.venv_path || path.join(ROOT_DIR, 'venv');
        const venv = getVenvPaths(venvPath);

        // Read config to check for GPU IDs
        let gpuEnv = '';
        let currentGpuIds = '';
        let accelerateFlags = '';
        try {
            const rawConfig = TOML.parse(fs.readFileSync(configPath, 'utf8'));
            currentGpuIds = rawConfig.gpu_ids ? rawConfig.gpu_ids.toString().trim() : '';
            const gpuIds = currentGpuIds;

            if (gpuIds) {
                // Validate format: "0", "0,1", "0, 1", etc.
                if (!/^[\d\s,]+$/.test(gpuIds)) {
                    return res.status(400).json({ error: `Invalid GPU IDs format: "${gpuIds}". Use numbers separated by commas (e.g. "0,1").` });
                }

                // Clean up and re-format
                const validIds = gpuIds.split(',')
                    .map(s => s.trim())
                    .filter(s => s.length > 0);

                // Ensure they are numbers
                if (validIds.some(id => isNaN(parseInt(id)))) {
                    return res.status(400).json({ error: "GPU IDs must be valid numbers." });
                }

                // Reconstruct safe string
                const safeGpuString = validIds.join(',');

                // Set explicit GPUs
                gpuEnv = buildEnvVar('CUDA_VISIBLE_DEVICES', safeGpuString);

                if (validIds.length > 1) {
                    accelerateFlags = `--multi_gpu --num_processes ${validIds.length}`;
                }
            }
        } catch (err) {
            console.warn("Failed to parse config for GPU options:", err);
            if (err.message && err.message.startsWith('Invalid GPU')) {
                return res.status(400).json({ error: err.message });
            }
        }

        // Spawn training process
        const trainEnvVars = [
            buildEnvVar('PYTHONIOENCODING', 'utf-8'),
            gpuEnv
        ].filter(Boolean).join('\n');
        const trainCmd = `python -m accelerate.commands.launch --num_cpu_threads_per_process 1 ${accelerateFlags} "${TRAINING_SCRIPT}" --config_file="${mergedConfigPath}"`;
        const trainScript = buildShellScript(venv.activate, trainEnvVars, trainCmd);

        const proc = spawnShell(trainScript, ROOT_DIR);

        const logBuffer = [];
        const MAX_LOG_LINES = 5000;

        // Write logs to file
        const logFileName = `train_${new Date().toISOString().replace(/[:.]/g, '-')}.log`;
        const logStream = fs.createWriteStream(path.join(logsDir, logFileName), { flags: 'a' });

        const appendLog = (data) => {
            const text = data.toString();
            logBuffer.push(text);
            if (logBuffer.length > MAX_LOG_LINES) logBuffer.shift();
            logStream.write(text);
            broadcastLog(jobName, text);
        };

        proc.stdout.on('data', appendLog);
        proc.stderr.on('data', appendLog);

        // Prevent crashes on stream errors
        proc.stdout.on('error', (err) => console.error(`[Train/stdout] ${err.message}`));
        proc.stderr.on('error', (err) => console.error(`[Train/stderr] ${err.message}`));
        logStream.on('error', (err) => console.error(`[Train/LogFile] ${err.message}`));

        proc.on('close', (code) => {
            const hints = getTrainingFailureHints(code, logBuffer);
            if (hints.length) {
                const hintBlock = `\n--- Crash diagnostics ---\n${hints.map(h => `- ${h}`).join('\n')}\n`;
                logStream.write(hintBlock);
                appendLog(Buffer.from(hintBlock));
            }

            const msg = `\n--- Training ${code === 0 ? 'completed' : 'stopped'} (exit code: ${code}) ---\n`;
            logStream.write(msg);
            logStream.end();
            appendLog(Buffer.from(msg));
            runningJobs.delete(jobName);
            broadcastStatus(jobName, 'idle');
        });

        proc.on('error', (err) => {
            appendLog(Buffer.from(`\nERROR: ${err.message}\n`));
            runningJobs.delete(jobName);
            broadcastStatus(jobName, 'idle');
        });

        runningJobs.set(jobName, {
            process: proc,
            pid: proc.pid,
            startTime: Date.now(),
            logBuffer,
            type: 'training',
            gpuIds: currentGpuIds
        });

        broadcastStatus(jobName, 'running');
        res.json({ success: true, pid: proc.pid });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});


app.get('/api/jobs/:name/train/status', (req, res) => {
    try {
        const jobName = sanitizeName(req.params.name);
        const isRunning = runningJobs.has(jobName);
        res.json({ running: isRunning });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// --- TensorBoard API ---

const tbProcesses = new Map(); // jobName -> { process, port }
let nextTbPort = 6006;

app.post('/api/jobs/:name/tensorboard', (req, res) => {
    try {
        const jobName = sanitizeName(req.params.name);
        const jobPath = getJobPath(jobName);
        const logsDir = path.join(jobPath, 'logs');

        if (tbProcesses.has(jobName)) {
            const tb = tbProcesses.get(jobName);
            return res.json({ success: true, port: tb.port, url: `http://localhost:${tb.port}` });
        }

        if (!fs.existsSync(logsDir)) {
            fs.mkdirSync(logsDir, { recursive: true });
        }

        // Get venv path
        const globalConfig = getGlobalConfig();
        const venvPath = globalConfig.venv_path || path.join(ROOT_DIR, 'venv');
        const venv = getVenvPaths(venvPath);

        const port = nextTbPort++;

        const tbCmd = `python -m tensorboard.main --logdir="${logsDir}" --port=${port} --host=0.0.0.0`;
        const tbScript = buildShellScript(venv.activate, '', tbCmd);
        const proc = spawnShell(tbScript, ROOT_DIR);

        proc.stderr.on('data', (data) => {
            const text = data.toString();
            console.log(`[TensorBoard/${jobName}] ${text.trim()}`);
        });

        proc.on('close', (code) => {
            console.log(`[TensorBoard/${jobName}] Exited with code ${code}`);
            tbProcesses.delete(jobName);
        });

        tbProcesses.set(jobName, { process: proc, port, pid: proc.pid });

        res.json({ success: true, port, url: `http://localhost:${port}` });

    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.post('/api/jobs/:name/tensorboard/stop', (req, res) => {
    try {
        const jobName = sanitizeName(req.params.name);
        const tb = tbProcesses.get(jobName);

        if (!tb) {
            return res.json({ success: true, message: 'Not running' });
        }

        if (tb.pid) {
            exec(`taskkill /PID ${tb.pid} /F /T`, (err) => {
                if (err) console.error(`TensorBoard taskkill failed: ${err}`);
            });
        }

        tbProcesses.delete(jobName);
        res.json({ success: true });

    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.get('/api/jobs/:name/tensorboard/status', (req, res) => {
    try {
        const jobName = sanitizeName(req.params.name);
        const tb = tbProcesses.get(jobName);
        if (tb) {
            res.json({ running: true, port: tb.port, url: `http://localhost:${tb.port}` });
        } else {
            res.json({ running: false });
        }
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// --- Samples API ---

function collectImages(dir, relBase, jobName) {
    const images = [];
    if (!fs.existsSync(dir)) return images;
    fs.readdirSync(dir, { withFileTypes: true }).forEach(entry => {
        const fullPath = path.join(dir, entry.name);
        if (entry.isDirectory()) {
            // Recurse into subdirectories (e.g. output/sample/)
            images.push(...collectImages(fullPath, path.join(relBase, entry.name), jobName));
        } else if (/\.(png|jpg|jpeg|webp)$/i.test(entry.name)) {
            const stat = fs.statSync(fullPath);
            const relPath = path.join(relBase, entry.name).replace(/\\/g, '/');
            images.push({
                name: entry.name,
                dir: relBase.replace(/\\/g, '/'),
                mtime: stat.mtimeMs,
                path: `/api/jobs/${jobName}/samples/${relPath}`
            });
        }
    });
    return images;
}

app.get('/api/jobs/:name/samples', (req, res) => {
    try {
        const jobPath = getJobPath(req.params.name);
        const samplesDir = path.join(jobPath, 'samples');
        const outputDir = path.join(jobPath, 'output');

        let images = [];
        images.push(...collectImages(samplesDir, 'samples', req.params.name));
        images.push(...collectImages(outputDir, 'output', req.params.name));

        images.sort((a, b) => b.mtime - a.mtime);
        res.set('Cache-Control', 'no-store');
        res.json(images);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Serve sample image files (supports nested paths like output/sample/img.png)
app.get('/api/jobs/:name/samples/*', (req, res) => {
    try {
        const jobPath = getJobPath(req.params.name);
        const relativePath = req.params[0]; // everything after /samples/
        const filePath = path.join(jobPath, relativePath);
        if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
            res.sendFile(filePath);
        } else {
            res.status(404).json({ error: 'File not found' });
        }
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Extract PNG Metadata helper
function extractPngMetadata(filePath) {
    try {
        const buffer = fs.readFileSync(filePath);
        let offset = 8; // skip PNG signature
        const metadata = {};

        while (offset < buffer.length) {
            const length = buffer.readUInt32BE(offset);
            const type = buffer.slice(offset + 4, offset + 8).toString('ascii');

            if (type === 'tEXt') {
                const data = buffer.slice(offset + 8, offset + 8 + length).toString('utf8');
                const nullIdx = data.indexOf('\u0000');
                if (nullIdx !== -1) {
                    const key = data.substring(0, nullIdx);
                    const val = data.substring(nullIdx + 1);
                    metadata[key] = val;
                }
            } else if (type === 'IEND') break;

            offset += 12 + length;
        }
        return metadata;
    } catch (e) {
        return null;
    }
}

// Get image metadata
app.get('/api/jobs/:name/metadata/*', (req, res) => {
    try {
        const jobPath = getJobPath(req.params.name);
        const relativePath = req.params[0];
        const filePath = path.join(jobPath, relativePath);

        if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
            const metadata = extractPngMetadata(filePath);
            res.json(metadata || {});
        } else {
            res.status(404).json({ error: 'File not found' });
        }
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Delete a sample image
app.delete('/api/jobs/:name/samples/*', (req, res) => {
    try {
        const jobPath = getJobPath(req.params.name);
        const relativePath = req.params[0];
        const filePath = path.join(jobPath, relativePath);

        if (fs.existsSync(filePath)) {
            fs.unlinkSync(filePath);
            res.json({ success: true });
        } else {
            res.status(404).json({ error: 'File not found' });
        }
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// --- Job Settings Actions ---

app.post('/api/jobs/:name/open-folder', (req, res) => {
    try {
        const jobPath = getJobPath(req.params.name);
        if (fs.existsSync(jobPath)) {
            if (process.platform === 'win32') {
                spawn('explorer', [jobPath]);
            } else {
                // Linux/Mac fallback (xdg-open or open)
                const cmd = process.platform === 'darwin' ? 'open' : 'xdg-open';
                spawn(cmd, [jobPath]);
            }
        }
        res.json({ success: true });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.post('/api/system/open-folder', (req, res) => {
    try {
        const { path: folderPath } = req.body;
        if (folderPath && fs.existsSync(folderPath)) {
            if (process.platform === 'win32') {
                spawn('explorer', [folderPath]);
            } else {
                const cmd = process.platform === 'darwin' ? 'open' : 'xdg-open';
                spawn(cmd, [folderPath]);
            }
            res.json({ success: true });
        } else {
            res.status(404).json({ error: 'Folder not found' });
        }
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.post('/api/jobs/:name/clear-logs', (req, res) => {
    try {
        const jobPath = getJobPath(req.params.name);
        const logsPath = path.join(jobPath, 'logs');
        if (fs.existsSync(logsPath)) {
            fs.readdirSync(logsPath).forEach(file => {
                const filePath = path.join(logsPath, file);
                try {
                    const stat = fs.statSync(filePath);
                    if (stat.isFile()) fs.unlinkSync(filePath);
                    else if (stat.isDirectory()) fs.rmSync(filePath, { recursive: true });
                } catch (e) { }
            });
        }
        res.json({ success: true });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.post('/api/jobs/:name/reset-config', (req, res) => {
    try {
        const jobPath = getJobPath(req.params.name);
        const { config } = getDefaultConfig();
        fs.writeFileSync(path.join(jobPath, 'config.toml'), TOML.stringify(config), 'utf8');
        const dataset = getDefaultDataset();
        fs.writeFileSync(path.join(jobPath, 'dataset.toml'), TOML.stringify(dataset), 'utf8');
        res.json({ success: true });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Prevent server crash on unhandled errors
process.on('uncaughtException', (err) => {
    console.error('CRITICAL ERROR (Uncaught Exception):', err);
});

process.on('unhandledRejection', (reason, promise) => {
    console.error('CRITICAL ERROR (Unhandled Rejection) at:', promise, 'reason:', reason);
});

// --- Start Server ---

function checkPort(port) {
    return new Promise((resolve) => {
        const tester = net.createServer();
        tester.once('error', () => resolve(false));
        tester.once('listening', () => {
            tester.close();
            resolve(true);
        });
        tester.listen(port);
    });
}

async function findAvailablePort(startPort, maxAttempts = 10) {
    for (let port = startPort; port < startPort + maxAttempts; port++) {
        if (await checkPort(port)) return port;
    }
    return null;
}

(async () => {
    const port = await findAvailablePort(DEFAULT_PORT);

    if (!port) {
        console.error(`\n❌ ERROR: No available port found in range ${DEFAULT_PORT}-${DEFAULT_PORT + 10}`);
        process.exit(1);
    }

    if (port !== DEFAULT_PORT) {
        console.warn(`⚠️ Port ${DEFAULT_PORT} was busy, using ${port} instead.`);
    }

    server.listen(port, () => {
        console.log(`🎯 Anima Training UI running at http://localhost:${port}`);
        try {
            if (process.platform === 'win32') {
                spawn('explorer', [`http://localhost:${port}`]);
            } else {
                const cmd = process.platform === 'darwin' ? 'open' : 'xdg-open';
                spawn(cmd, [`http://localhost:${port}`]);
            }
        } catch (e) {
            console.warn('Could not open browser automatically.');
        }
    });
})();

