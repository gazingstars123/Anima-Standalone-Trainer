[IMPORTANT]
> ## Anima 2.9B LoRA training is supported

# Anima Standalone Trainer

A lightweight, decoupled training environment for circlestone-labs' Anima model, currently support Lora training only. Windows and Linux support. Built upon [sd-scripts](https://github.com/kohya-ss/sd-scripts) implementation.

<img width="2554" height="1234" alt="image" src="https://github.com/user-attachments/assets/cb5ff930-ce8c-49d6-a77a-3da393fe719d" />


## Prerequisites

- **Python 3.10+** (Python 3.12 recommended)
- **Node.js** (Required for the Web UI)
- **CUDA fitting your system** (CUDA 12.7+ recommended)

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/gazingstars123/Anima-Standalone-Trainer.git
cd Anima-Standalone-Trainer
```

### 2. Set up the environment

Run the provided setup script for your operating system:

**Windows:**
```powershell
.\setup_env.bat
```

**Linux:**
```bash
./setup_env.sh
```

*This will create a virtual environment (`venv`), install all Python dependencies (assuming you have met the prereqisites), and set up the Web UI.*

This script will probably install Torch and Torchvision version below.
Depends on your system, you may want to install another version of Pytorch with CUDA.

```cmd
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

## Launching the UI

To start the training server and open the web interface:

**Windows:**
```cmd
.\training-ui\start_training_ui_anima.bat
```

**Linux:**
```bash
./training-ui/start_linux.sh
```
Once launched, open your browser to: `http://localhost:3000`

## First Time Setup

After launching the UI for the first time, you'll need to configure your model paths:

1. Click the ** Global Settings** (gear icon) in the bottom-left corner
2. Set the following paths:
   - **DiT Model Path** — Path to your Anima DiT safetensors file (e.g. `C:\model\anima.safetensors`)
   - **VAE Path** — Path to the VAE model (e.g. `C:\model\qwen_image_vae.safetensors`)  
   - **TE Path** — Path to the CLIP text encoder (e.g. `C:\model\text_encoders\qwen_3_06b_base.safetensors`)
   - **Venv Path** - Path to your local venv, venv can be reused if you redownload the repo
3. Click **Save**

These paths are saved globally and shared across all training jobs.

## Release

**v2.0.0. Linux support, Multi-GPU inference**

**v1.1.0. Improving caching and others I/O performance.**

## Multi-GPU

Tested on torch2.7+cu128 and torch2.10+cu130 with [this fix](https://github.com/pytorch/pytorch/pull/175316) applied on Windows when encountered **libuv** error.

Seems to works best with torch<=2.3 and cuda <= 12.4 without directly applying the fix.

\**NEW\**

Adding support for multi-gpu inference

<img width="1052" height="848" alt="image" src="https://github.com/user-attachments/assets/54192c8f-1501-4a38-b745-3b26499aca5f" />


## Update

To update, simply run this command

```cmd
git pull
```

## Misc

Some features and settings from sd-scripts may not be available or working properly at the momment.

Built and tested on Windows 11, RTX 5080 + RTX 3090, 96GB DDR5, Python 3.12.1, CUDA 13.1, Pytorch 2.10 


