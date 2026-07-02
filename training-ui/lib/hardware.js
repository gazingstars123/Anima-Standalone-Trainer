const { spawn } = require('child_process');
const os = require('os');
const WebSocket = require('ws');

const isWindows = process.platform === 'win32';
const isWSL = process.platform === 'linux' && !!process.env.WSL_DISTRO_NAME;
const HW_MONITOR_INTERVAL_MS = 1000;

let prevCpuInfo = null;

function getCpuUsagePct() {
    const cpus = os.cpus();
    if (!prevCpuInfo) {
        prevCpuInfo = cpus;
        return 0;
    }
    let totalDelta = 0, idleDelta = 0;
    cpus.forEach((cpu, i) => {
        const prev = prevCpuInfo[i];
        if (!prev) return;
        const prevTotal = Object.values(prev.times).reduce((a, b) => a + b, 0);
        const currTotal = Object.values(cpu.times).reduce((a, b) => a + b, 0);
        totalDelta += currTotal - prevTotal;
        idleDelta += cpu.times.idle - prev.times.idle;
    });
    prevCpuInfo = cpus;
    if (totalDelta === 0) return 0;
    return Math.round((1 - idleDelta / totalDelta) * 100);
}

function terminateChildProcess(child, { force = false } = {}) {
    if (!child || child.pid == null) return;

    if (isWindows) {
        const args = ['/PID', String(child.pid), '/T'];
        if (force) args.splice(2, 0, '/F');
        const killer = spawn('taskkill', args, { stdio: 'ignore', windowsHide: true });
        killer.on('error', () => { });
        return;
    }

    const signal = force ? 'SIGKILL' : 'SIGTERM';
    try {
        process.kill(child.pid, signal);
    } catch (_) {
        try { process.kill(-child.pid, signal); } catch (__) { }
    }
}

function runSingleFlightProbe(state, spawnChild, timeoutMs, parseResult) {
    if (state.pending) return Promise.resolve(null);
    state.pending = true;

    return new Promise((resolve) => {
        const child = spawnChild();
        let stdout = '';
        let settled = false;
        let timedOut = false;
        let forceKillTimer = null;

        const finish = (value) => {
            if (settled) return;
            settled = true;
            state.pending = false;
            clearTimeout(timeout);
            if (forceKillTimer) clearTimeout(forceKillTimer);
            resolve(value);
        };

        const timeout = setTimeout(() => {
            timedOut = true;
            terminateChildProcess(child);
            forceKillTimer = setTimeout(() => terminateChildProcess(child, { force: true }), 1500);
        }, timeoutMs);

        child.stdout.on('data', (data) => {
            stdout += data;
        });

        child.on('close', (code) => {
            if (timedOut || code !== 0) {
                finish(null);
                return;
            }
            try {
                finish(parseResult(stdout));
            } catch (_) {
                finish(null);
            }
        });

        child.on('error', () => {
            finish(null);
        });
    });
}

const cpuTempProbeState = { pending: false };
const gpuStatsProbeState = { pending: false };

let gpuSmiBinary = 'nvidia-smi';

function probeWslSmiExe() {
    if (!isWSL) return;
    const probe = spawn('nvidia-smi.exe', ['--query-gpu=index', '--format=csv,noheader'], { windowsHide: true });
    let stdout = '';
    const timeout = setTimeout(() => terminateChildProcess(probe, { force: true }), 4000);
    probe.stdout.on('data', (data) => { stdout += data; });
    probe.on('close', (code) => {
        clearTimeout(timeout);
        if (code === 0 && stdout.trim()) gpuSmiBinary = 'nvidia-smi.exe';
    });
    probe.on('error', () => { clearTimeout(timeout); });
}

function getCpuTemp() {
    return runSingleFlightProbe(
        cpuTempProbeState,
        () => {
            if (isWindows) {
                return spawn('powershell', [
                    '-NoProfile', '-NonInteractive', '-Command',
                    'Get-WmiObject -Namespace root/wmi -Class MSAcpi_ThermalZoneTemperature | Select-Object -ExpandProperty CurrentTemperature'
                ], { windowsHide: true });
            }

            return spawn('bash', ['-c',
                'paste -sd+ /sys/class/thermal/thermal_zone*/temp 2>/dev/null | bc'
            ], { windowsHide: true });
        },
        4000,
        (stdout) => {
            if (!stdout.trim()) return null;

            if (isWindows) {
                const vals = stdout.trim().split('\n')
                    .map(l => parseFloat(l.trim()))
                    .filter(v => !isNaN(v) && v > 0);
                if (!vals.length) return null;
                return Math.round(vals.reduce((a, b) => a + b, 0) / vals.length / 10 - 273.15);
            }

            const val = parseFloat(stdout.trim());
            return isNaN(val) ? null : Math.round(val / 1000);
        }
    );
}

function getGpuStats() {
    return runSingleFlightProbe(
        gpuStatsProbeState,
        () => spawn(gpuSmiBinary, [
            '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit',
            '--format=csv,noheader,nounits'
        ], { windowsHide: true }),
        3000,
        (stdout) => {
            if (!stdout.trim()) return null;

            return stdout.trim().split('\n').map(line => {
                const parts = line.split(',').map(s => s.trim());
                return {
                    index: parseInt(parts[0]),
                    name: parts[1],
                    util: parseInt(parts[2]) || 0,
                    memUsed: parseInt(parts[3]) || 0,
                    memTotal: parseInt(parts[4]) || 0,
                    temp: parseInt(parts[5]) || 0,
                    powerDraw: Math.round(parseFloat(parts[6])) || 0,
                    powerLimit: Math.round(parseFloat(parts[7])) || 0
                };
            });
        }
    );
}

// wss       - WebSocket.Server instance
// getActiveGpus - () => { [gpuIndex]: 'training'|'sampling' }
function startHardwareMonitor(wss, getActiveGpus) {
    probeWslSmiExe();
    setInterval(async () => {
        if (wss.clients.size === 0) return;

        const cpuPct = getCpuUsagePct();
        const totalMem = os.totalmem();
        const freeMem = os.freemem();
        const [gpus, cpuTemp] = await Promise.all([getGpuStats(), getCpuTemp()]);

        if (gpus === null) return;

        const activeGpus = getActiveGpus();
        gpus.forEach(gpu => {
            gpu.activity = activeGpus[String(gpu.index)] || null;
        });

        const payload = JSON.stringify({
            type: 'hw_stats',
            data: {
                cpu: cpuPct,
                cpuTemp,
                ram: { total: totalMem, used: totalMem - freeMem },
                gpus
            }
        });

        wss.clients.forEach(client => {
            if (client.readyState === WebSocket.OPEN) {
                client.send(payload);
            }
        });
    }, HW_MONITOR_INTERVAL_MS);
}

module.exports = { startHardwareMonitor };
