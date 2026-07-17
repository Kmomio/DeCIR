# Installation Guide

## Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- 24GB+ GPU memory recommended

## Quick Install

### 1. Clone Repository

```bash
git clone https://github.com/your-org/DeCIR.git
cd DeCIR
```

### 2. Create Environment

**Option A: Conda (Recommended)**

```bash
conda env create -f environment.yml
conda activate decir
```

**Option B: pip + venv**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Install DeCIR Package

```bash
pip install -e .
```

## Model Downloads

DeCIR uses the following pre-trained models (auto-downloaded on first run):

- **Qwen3-VL-8B**: `Qwen/Qwen2-VL-7B-Instruct` (~15GB)
- **CLIP-ViT-L-14**: `laion/CLIP-ViT-L-14-laion2B-s32B-b82K` (~1.7GB)
- **SDXL Inpainting**: `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` (~10GB)

Total: ~27GB

## Verification

Test your installation:

```bash
python -c "import decir; print(decir.__version__)"
```

You should see: `0.1.0`

## Troubleshooting

### CUDA Out of Memory

If you encounter OOM errors:
1. Reduce batch size
2. Use mixed precision (default: fp16)
3. Enable gradient checkpointing

### Model Download Issues

If Hugging Face downloads fail:
```bash
export HF_ENDPOINT=https://hf-mirror.com  # Use mirror
# or
huggingface-cli login  # Authenticate
```

### Dependencies Conflicts

If you encounter dependency conflicts:
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --no-deps
pip check  # Verify installation
```

## Next Steps

See [Quick Start Guide](quickstart.md) for usage examples.
