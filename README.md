# DeCIR: Dual-modal Semantic Decoupling for Composed Image Retrieval

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**A Training-Free Zero-Shot Framework for Composed Image Retrieval**

[Paper (Coming Soon)] | [Demo (Coming Soon)] | [Documentation](docs/)

</div>

---

## 📖 Abstract

**Composed Image Retrieval (CIR)** is the task of retrieving target images that preserve the visual content of a reference image while incorporating user-specified textual modifications. Existing approaches often require expensive task-specific training or struggle to accurately capture nuanced user intent.

We present **DeCIR**, a novel **training-free zero-shot framework** that introduces **Dual-modal Semantic Decoupling** to address these challenges. Our key innovation is a multi-stage pipeline that:

1. **Decouples semantic changes from global image features** through precise visual grounding and localized editing
2. **Fuses multi-modal signals** via an adaptive weighting mechanism: `E_query = α·E_image + β·E_text + γ·E_delta`
3. **Achieves strong zero-shot performance** without requiring any task-specific training or labeled data

DeCIR leverages pre-trained vision-language models (Qwen3-VL, CLIP, SDXL) in a novel architecture that maintains high performance across diverse CIR benchmarks while being extremely efficient and adaptable.

---

## 🌟 Key Contributions

### 1. Dual-modal Semantic Decoupling Architecture

We propose a systematic approach to decompose the composed image retrieval task into interpretable stages:

- **Intent Analysis**: Vision-language model extracts structured edit intent from modification text
- **Visual Grounding**: Precise localization of target objects and regions
- **Localized Editing**: Semantic-aware mask generation and SDXL-based inpainting
- **Dual-modal Fusion**: Novel fusion of global image, text semantics, and visual deltas

### 2. Visual Delta Extraction

A core innovation is our **visual delta computation**:

```
E_delta = E(edited_patch) - E(original_patch)
```

This captures the **semantic difference** introduced by the modification while filtering out irrelevant global context, leading to more precise retrieval.

### 3. Training-Free Zero-Shot Paradigm

- ✅ **No training required** - directly applicable to any CIR dataset
- ✅ **No fine-tuning needed** - leverages pre-trained models as-is
- ✅ **Zero labeled data** - works in pure zero-shot setting
- ✅ **Fast adaptation** - can be deployed to new domains immediately

### 4. Extensive Benchmarking

Strong performance demonstrated on four standard CIR benchmarks:
- **CIRR** (Composed Image Retrieval on Real-life images)
- **CIRCO** (Open-domain composed image retrieval)
- **FashionIQ** (Fashion-specific CIR)
- **GeneCIS** (Generalized composed image search)

---

## 🏗️ Architecture Overview

<div align="center">

```
Reference Image + Modification Text
            ↓
┌───────────────────────────────────────────────────────────┐
│  Intent Parser                                            │
│  → Structured JSON: edit_type, operations, targets        │
└───────────────────────────────────────────────────────────┘
            ↓
┌───────────────────────────────────────────────────────────┐
│  Visual Grounding                                         │
│  → Bounding boxes of target objects                       │
└───────────────────────────────────────────────────────────┘
            ↓
┌───────────────────────────────────────────────────────────┐
│  Region Caption & Target Rewrite                          │
│  → Target descriptions for editing                        │
└───────────────────────────────────────────────────────────┘
            ↓
┌───────────────────────────────────────────────────────────┐
│  Mask Engine                                              │
│  → Semantic-aware edit masks                              │
└───────────────────────────────────────────────────────────┘
            ↓
┌───────────────────────────────────────────────────────────┐
│  Image Editor                                             │
│  → Edited image with localized modifications              │
└───────────────────────────────────────────────────────────┘
            ↓
┌───────────────────────────────────────────────────────────┐
│  Query Builder (CLIP)                                     │
│                                                           │
│  E_query = α·E_image + β·E_text + γ·E_delta               │
│                                                           │
│  where E_delta = E(edit_patch) - E(ref_patch)             │
└───────────────────────────────────────────────────────────┘
            ↓
    Retrieval Query Embedding
```

</div>

---

## 📂 Project Structure

```
DeCIR/                                      # 项目根目录 / Project root
├── README.md                               # 项目说明文档 / Project documentation
├── LICENSE                                 # MIT开源协议 / MIT License
├── setup.py                                # Python包安装脚本 / Package installation script
├── requirements.txt                        # Python依赖列表 / Python dependencies
├── environment.yml                         # Conda环境配置 / Conda environment config
├── .gitignore                              # Git忽略规则 / Git ignore rules
├── PROJECT_SUMMARY.md                      # 项目摘要 / Project summary
│
├── decir/                                  # 核心代码包（可导入）/ Core package (importable)
│   ├── __init__.py                         # 包初始化文件 / Package initializer
│   │
│   ├── core/                               # 核心算法模块 / Core algorithms
│   │   ├── __init__.py
│   │   ├── pipeline.py                     # 端到端流水线 / End-to-end pipeline
│   │   ├── stages/                         # 双模态解耦各阶段 / Dual-modal decoupling stages
│   │   │   ├── __init__.py
│   │   │   ├── intent_parser.py            # Stage 1: 意图解析 / Intent parsing
│   │   │   ├── visual_grounding.py         # Stage 2A: 视觉定位 / Visual grounding
│   │   │   ├── region_caption.py           # Stage 2B: 区域描述 / Region captioning
│   │   │   ├── target_rewrite.py           # Stage 2.5: 目标重写 / Target rewriting
│   │   │   ├── mask_engine.py              # Stage 3: 掩码生成 / Mask generation
│   │   │   ├── image_edit.py               # Stage 4: 图像编辑（SDXL）/ Image editing (SDXL)
│   │   │   └── query_builder.py            # Stage 9: 查询构建（核心融合）/ Query builder (core fusion)
│   │   └── utils/                          # 核心工具模块 / Core utilities
│   │       └── __init__.py
│   │
│   ├── models/                             # 预训练模型封装 / Pre-trained model wrappers
│   │   ├── __init__.py
│   │   ├── qwen_client.py                  # Qwen3-VL客户端 / Qwen3-VL client
│   │   ├── clip_encoder.py                 # CLIP编码器 / CLIP encoder
│   │   └── sdxl_client.py                  # SDXL客户端 / SDXL client
│   │
│   ├── datasets/                           # 数据集加载器 / Dataset loaders
│   │   ├── __init__.py
│   │   ├── base.py                         # 基类 / Base class
│   │   └── cirr.py                         # CIRR数据集加载 / CIRR dataset loader
│   │
│   ├── retrieval/                          # 检索与重排序 / Retrieval & reranking
│   │   ├── __init__.py
│   │   ├── inference.py                    # 推理接口 / Inference interface
│   │   └── reranking.py                    # MLLM重排序 / MLLM reranking
│   │
│   └── utils/                              # 通用工具函数 / General utilities
│       ├── __init__.py
│       ├── image_ops.py                    # 图像操作 / Image operations
│       ├── patch_ops.py                    # 视觉补丁提取 / Visual patch extraction
│       ├── intent_schema.py                # Intent JSON Schema定义 / Intent schema
│       └── logging.py                      # 日志工具 / Logging utilities
│
├── scripts/                                # 运行脚本 / Execution scripts
│   ├── inference/                          # 推理脚本 / Inference scripts
│   │   └── run_single_sample.py            # 单样本推理示例 / Single sample inference
│   ├── preprocessing/                      # 数据预处理 / Data preprocessing
│   ├── reranking/                          # 重排序脚本 / Reranking scripts
│   └── evaluation/                         # 评估脚本 / Evaluation scripts
│
├── configs/                                # 配置文件 / Configuration files
│   └── default.yaml                        # 默认配置（融合权重、模型路径等）/ Default config
│
├── prompts/                                # Prompt模板库 / Prompt templates
│   ├── vis_dec_prompt.txt                  # 视觉解耦Prompt / Visual decoupling prompt
│   ├── cirrCaptionPrompt.txt               # CIRR数据集Caption / CIRR caption
│   ├── circoCaptionPrompt.txt              # CIRCO数据集Caption / CIRCO caption
│   ├── fashioniqCaptionPrompt.txt          # FashionIQ数据集Caption / FashionIQ caption
│   ├── genecisCaptionPrompt.txt            # GeneCIS数据集Caption / GeneCIS caption
│   ├── cirr_caption_qwen35.txt             # Qwen3.5专用Caption / Qwen3.5 caption
│   └── rerank.txt                          # 重排序Prompt / Reranking prompt
│
├── docs/                                   # 项目文档 / Documentation
│   ├── installation.md                     # 安装指南 / Installation guide
│   └── quickstart.md                       # 快速开始 / Quick start guide
│
├── tests/                                  # 单元测试（待完善）/ Unit tests (to be implemented)
│
├── assets/                                 # 资源文件 / Assets
│   └── demo_images/                        # 演示图片 / Demo images
│
└── data/                                   # 数据集目录（.gitignore已忽略）/ Datasets (ignored by .gitignore)
    ├── cirr/                               # CIRR数据集 / CIRR dataset
    │   ├── dev/                            # 开发集 / Dev set
    │   ├── test1/                          # 测试集 / Test set
    │   ├── image_splits/                   # 图像集合 / Image collections
    │   └── cirr_dataset.json               # 数据集标注 / Dataset annotations
    │
    ├── circo/                              # CIRCO数据集 / CIRCO dataset
    │   ├── annotations/                    # 标注文件 / Annotations
    │   └── COCO2017_unlabeled/             # COCO图像 / COCO images
    │
    ├── fashioniq/                          # FashionIQ数据集 / FashionIQ dataset
    │   ├── images/                         # 图像 / Images
    │   ├── captions/                       # 描述 / Captions
    │   └── image_splits/                   # 数据集划分 / Dataset splits
    │       ├── split.dress.*.json          # Dress类别 / Dress category
    │       ├── split.shirt.*.json          # Shirt类别 / Shirt category
    │       └── split.toptee.*.json         # Toptee类别 / Toptee category
    │
    └── genecis/                            # GeneCIS数据集 / GeneCIS dataset
        ├── val/                            # 验证集 / Validation set
        ├── test/                           # 测试集 / Test set
        └── annotations/                    # 标注文件 / Annotations
```

### 📋 Key Directories

**Core Algorithm Implementation:**
- `decir/core/stages/` - Contains all stages of the dual-modal semantic decoupling pipeline
- `decir/core/stages/query_builder.py` - Implements the core fusion formula: `E_query = α·E_image + β·E_text + γ·E_delta`

**Model Wrappers:**
- `decir/models/` - Unified interfaces for pre-trained models (Qwen3-VL, CLIP, SDXL)

**Datasets:**
- `data/` - Dataset directory (excluded from git, see `.gitignore`)
- Download instructions for each dataset are provided in the "Benchmark Datasets" section below

**Configuration:**
- `configs/default.yaml` - Central configuration for fusion weights (α, β, γ), model paths, and hyperparameters
- `prompts/` - All prompt templates for reproducibility and tuning

---

## 🚀 Quick Start

### Installation

**Prerequisites:**
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- 24GB+ GPU memory recommended

**Step 1: Clone Repository**

```bash
git clone https://github.com/your-org/DeCIR.git
cd DeCIR
```

**Step 2: Create Environment**

```bash
# Option A: Using conda (recommended)
conda env create -f environment.yml
conda activate decir

# Option B: Using pip
pip install -r requirements.txt
```

**Step 3: Install Package**

```bash
pip install -e .
```

### Models

DeCIR uses the following pre-trained models (auto-downloaded on first run):

| Model | Size | Purpose |
|-------|------|---------|
| [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) | ~15GB | Intent parsing & visual grounding |
| [CLIP-ViT-L-14](https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K) | ~1.7GB | Image-text retrieval |
| [SDXL-Inpainting](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) | ~10GB | Localized image editing |

**Total:** ~27GB

---

## 💡 Usage

### Single Image Inference

```bash
python scripts/inference/run_single_sample.py \
    --reference_image examples/car.jpg \
    --modification_text "change the car color to blue" \
    --output_dir outputs/demo \
    --alpha 0.6 --beta 0.2 --gamma 0.2
```

**Output:**
- `query_embedding.npy` - Retrieval query embedding
- `edited_image.png` - Edited image (intermediate result)
- `intent.json` - Parsed edit intent
- `mask_*.png` - Generated edit masks

### Python API

```python
from decir import DeCIRPipeline
from decir.models import Qwen3VLClient, CLIPEncoder, SDXLInpaintClient
from decir.models.clip_encoder import CLIPConfig
from decir.models.sdxl_client import SDXLConfig

# Initialize models
qwen = Qwen3VLClient(model_path="Qwen/Qwen2-VL-7B-Instruct")
clip = CLIPEncoder(CLIPConfig(model_name="laion/CLIP-ViT-L-14"))
sdxl = SDXLInpaintClient(SDXLConfig())

# Create pipeline
pipeline = DeCIRPipeline(
    qwen_client=qwen,
    clip_encoder=clip,
    sdxl_client=sdxl,
    alpha=0.6,  # Reference image weight
    beta=0.2,   # Modification text weight
    gamma=0.2   # Visual delta weight
)

# Generate query embedding
query_emb = pipeline(
    reference_image="path/to/image.jpg",
    modification_text="your modification text"
)

# Retrieve with CLIP similarity
# similarities = np.dot(gallery_embeddings, query_emb)
# top_results = np.argsort(similarities)[::-1][:50]
```

### Fast Testing (Mock Mode)

For quick testing without loading heavy models:

```bash
python scripts/inference/run_single_sample.py \
    --reference_image test.jpg \
    --modification_text "test" \
    --mock_sdxl \
    --output_dir outputs/test
```

---

## 📊 Benchmark Datasets

### CIRR Dataset

```bash
# Download from: https://github.com/Cuberick-Orion/CIRR
mkdir -p data/cirr
# Extract to data/cirr/

# Run inference
python scripts/inference/run_cirr.py \
    --data_root data/cirr \
    --split test \
    --output_dir outputs/cirr
```

### CIRCO Dataset

```bash
# Download from: https://github.com/miccunifi/CIRCO
mkdir -p data/circo
# Extract to data/circo/

# Run inference
python scripts/inference/run_circo.py \
    --data_root data/circo \
    --output_dir outputs/circo
```

### FashionIQ Dataset

```bash
# Download from: https://github.com/XiaoxiaoGuo/fashion-iq
mkdir -p data/fashioniq
# Extract to data/fashioniq/

# Run inference (per category)
for category in dress shirt toptee; do
    python scripts/inference/run_fashioniq.py \
        --data_root data/fashioniq \
        --category $category \
        --output_dir outputs/fashioniq_$category
done
```

### GeneCIS Dataset

```bash
# Download from: https://github.com/facebookresearch/genecis
mkdir -p data/genecis
# Extract to data/genecis/

# Run inference
python scripts/inference/run_genecis.py \
    --data_root data/genecis \
    --output_dir outputs/genecis
```

---

## 🔬 Paper Reproduction Guide

This section provides a **complete step-by-step guide** to reproduce the results from our paper.

### Prerequisites

Before starting, ensure you have:
- ✅ Completed installation (see [Quick Start](#-quick-start))
- ✅ GPU with 24GB+ VRAM (tested on A100/V100)
- ✅ ~100GB free disk space for datasets
- ✅ Stable internet connection for model downloads

### Step 1: Prepare Datasets

#### CIRR Dataset

```bash
# 1. Download CIRR dataset
cd data/
git clone https://github.com/Cuberick-Orion/CIRR.git cirr_raw
cd cirr_raw
# Follow their instructions to download images and annotations

# 2. Organize files (expected structure)
cd ../
mkdir -p cirr
mv cirr_raw/dev cirr/
mv cirr_raw/test1 cirr/
mv cirr_raw/cirr cirr/

# 3. Verify structure
ls -R cirr/
# Expected output:
# cirr/
# ├── dev/              (~10K images)
# ├── test1/            (~5K images)
# └── cirr/
#     └── captions/
#         ├── cap.rc2.train.json
#         ├── cap.rc2.val.json
#         └── cap.rc2.test1.json
```

**Dataset size:** ~8GB

#### CIRCO Dataset

```bash
cd data/
mkdir -p circo

# 1. Download COCO2017 unlabeled images (CIRCO uses COCO images)
wget http://images.cocodataset.org/zips/unlabeled2017.zip
unzip unlabeled2017.zip -d circo/
mv circo/unlabeled2017 circo/COCO2017_unlabeled

# 2. Download CIRCO annotations
git clone https://github.com/miccunifi/CIRCO.git circo_anno
cp -r circo_anno/annotations circo/

# 3. Verify
ls circo/
# Expected:
# - COCO2017_unlabeled/ (~120K images)
# - annotations/
```

**Dataset size:** ~19GB

#### FashionIQ Dataset

```bash
cd data/
mkdir -p fashioniq

# 1. Download from official source
# Visit: https://github.com/XiaoxiaoGuo/fashion-iq
# Download images.tar.gz and captions.tar.gz

# 2. Extract
tar -xzf images.tar.gz -C fashioniq/
tar -xzf captions.tar.gz -C fashioniq/

# 3. Verify structure
ls fashioniq/
# Expected:
# - images/
# - captions/
# - image_splits/
```

**Dataset size:** ~30GB

#### GeneCIS Dataset

```bash
cd data/
mkdir -p genecis

# Download from: https://github.com/facebookresearch/genecis
# Follow their instructions

# Expected structure:
# genecis/
# ├── val/
# ├── test/
# └── annotations/
```

**Dataset size:** ~5GB

---

### Step 2: Extract Gallery Features

**Why this step?** For efficient retrieval, we pre-compute CLIP embeddings for all gallery (candidate) images.

```bash
# CIRR gallery
python scripts/preprocessing/extract_gallery_features.py \
    --dataset cirr \
    --data_root data/cirr \
    --split test \
    --output data/cirr/gallery_embeddings_test.npy \
    --batch_size 64

# CIRCO gallery
python scripts/preprocessing/extract_gallery_features.py \
    --dataset circo \
    --data_root data/circo \
    --output data/circo/gallery_embeddings.npy \
    --batch_size 64
```

**Expected outputs:**
- `data/cirr/gallery_embeddings_test.npy` (~50MB, shape: [N, 768])
- `data/circo/gallery_embeddings.npy` (~1GB, shape: [~120K, 768])

**Time estimate:** 10-30 minutes per dataset (depending on GPU)

---

### Step 3: Run Full Inference

#### CIRR Inference

```bash
python scripts/inference/run_cirr.py \
    --data_root data/cirr \
    --split test \
    --gallery_embeddings data/cirr/gallery_embeddings_test.npy \
    --output_dir outputs/cirr_test \
    --alpha 0.6 --beta 0.2 --gamma 0.2 \
    --batch_size 8
```

**Outputs:**
- `outputs/cirr_test/query_embeddings.npy` - All query embeddings
- `outputs/cirr_test/predictions.json` - Top-50 predictions per query
- `outputs/cirr_test/submission.json` - Formatted for CIRR evaluation server

**Time estimate:** 2-4 hours for full CIRR test set (~4K queries)

#### CIRCO Inference

```bash
python scripts/inference/run_circo.py \
    --data_root data/circo \
    --gallery_embeddings data/circo/gallery_embeddings.npy \
    --output_dir outputs/circo \
    --alpha 0.5 --beta 0.3 --gamma 0.2
```

#### FashionIQ Inference (per category)

```bash
for category in dress shirt toptee; do
    python scripts/inference/run_fashioniq.py \
        --data_root data/fashioniq \
        --category $category \
        --split test \
        --output_dir outputs/fashioniq_${category} \
        --alpha 0.7 --beta 0.15 --gamma 0.15
done
```

---

### Step 4: Evaluate Results

#### CIRR Evaluation

```bash
python scripts/evaluation/evaluate_cirr.py \
    --data_root data/cirr \
    --predictions outputs/cirr_test/predictions.json \
    --split test

# Expected output:
# ========================================
# CIRR Test Set Results
# ========================================
# Recall@1:  XX.XX%
# Recall@5:  XX.XX%
# Recall@10: XX.XX%
# Recall@50: XX.XX%
# ========================================
```

#### Submit to CIRR Server (Optional)

```bash
# Submit outputs/cirr_test/submission.json to:
# https://cirr.cecs.anu.edu.au/test_process/
```

#### FashionIQ Evaluation

```bash
python scripts/evaluation/evaluate_fashioniq.py \
    --predictions outputs/fashioniq_dress/predictions.json \
    --predictions outputs/fashioniq_shirt/predictions.json \
    --predictions outputs/fashioniq_toptee/predictions.json

# Expected output:
# ========================================
# FashionIQ Results (Average over 3 categories)
# ========================================
# Recall@10:  XX.XX%
# Recall@50:  XX.XX%
# ========================================
```

---

### Step 5: (Optional) MLLM Re-ranking

For improved results, apply MLLM re-ranking on top-K candidates:

```bash
python scripts/reranking/rerank_cirr.py \
    --data_root data/cirr \
    --predictions outputs/cirr_test/predictions.json \
    --top_k 10 \
    --output outputs/cirr_test/reranked_predictions.json

# Then re-evaluate
python scripts/evaluation/evaluate_cirr.py \
    --predictions outputs/cirr_test/reranked_predictions.json
```

**Note:** Re-ranking is compute-intensive (~5-10 seconds per query) but can boost Recall@1 by 2-3%.

---

### Expected Performance

Our paper reports the following results:

#### CIRR Test Set

Results with different CLIP backbones:

| Backbone | Method | R@1 | R@5 | R@10 | R@50 | RecallSubset@1 | RecallSubset@2 | RecallSubset@3 |
|----------|--------|-----|-----|------|------|----------------|----------------|----------------|
| ViT-B/32 | DeCIR (ours) | 44.54 | 71.03 | 79.41 | 91.93 | 77.82 | 89.92 | 93.40 |
| ViT-L/14 | DeCIR (ours) | 45.07 | 72.55 | 80.21 | 92.81 | 78.30 | 90.63 | 95.27 |
| ViT-G/14 | DeCIR (ours) | **45.52** | **73.35** | **80.82** | **93.76** | **79.16** | **91.54** | **96.16** |

**Note:** CIRR has been found to suffer from modality shortcut learning [47, 48]. DeCIR intentionally avoids text-dominant shortcuts by strictly preserving visual priors through semantic decoupling. Therefore, DeCIR's true superiority is more accurately reflected in the RecallSubset metrics, which strictly demand fine-grained visual preservation.

#### CIRCO Test Set

Results with different CLIP backbones:

| Backbone | Method | mAP@5 | mAP@10 | mAP@25 | mAP@50 |
|----------|--------|-------|--------|--------|--------|
| ViT-B/32 | DeCIR (ours) | 35.17 | 36.21 | 37.38 | 37.94 |
| ViT-L/14 | DeCIR (ours) | 38.79 | 39.18 | 40.53 | 41.05 |
| ViT-G/14 | DeCIR (ours) | **41.15** | **41.27** | **41.75** | **42.81** |

#### FashionIQ (Average over 3 categories)

Results with different CLIP backbones:

| Backbone | Method | R@10 | R@50 |
|----------|--------|------|------|
| ViT-B/32 | DeCIR (ours) | 39.91 | 60.81 |
| ViT-L/14 | DeCIR (ours) | 42.32 | 62.22 |
| ViT-G/14 | DeCIR (ours) | **45.32** | **64.14** |

**Key Achievements:**
- **CIRCO**: Outperforms WISER by +4.62% in mAP@5 (ViT-G/14)
- **CIRR**: Achieves +2.48% and +3.48% boost in RecallSubset@2 and RecallSubset@3 (ViT-G/14)
- **FashionIQ**: Improves Recall@50 by +3.71% compared to WISER (ViT-L/14), even outperforms supervised LinCIR by +0.21% in Recall@10 (ViT-G/14)

**Note:** If you reproduce results within ±1-2% of these numbers, the reproduction is successful!

---

### Troubleshooting

#### Issue 1: Out of GPU Memory

**Solution:**
```bash
# Reduce batch size
--batch_size 4  # or even 2

# Use mock SDXL for testing
--mock_sdxl

# Or use gradient checkpointing (modify code)
```

#### Issue 2: Models fail to download from Hugging Face

**Solution:**
```bash
# Use mirror (China users)
export HF_ENDPOINT=https://hf-mirror.com

# Or manually download models to local cache
# Set environment variable
export TRANSFORMERS_CACHE=/path/to/your/model/cache
```

#### Issue 3: CIRR evaluation script fails

**Solution:**
```bash
# Ensure prediction format matches:
{
  "query_id": "test1-0-0",
  "ranking": ["img1", "img2", ..., "img50"]
}
```

#### Issue 4: Slow inference (>10s per query)

**Expected behavior:** Each query should take ~2-5 seconds on A100.

**Solution:**
- Check GPU utilization: `nvidia-smi`
- Ensure CUDA is properly installed
- Disable interactive mask expansion for speed

---

### Reproducibility Checklist

Before claiming successful reproduction, verify:

- ✅ All datasets downloaded and structured correctly
- ✅ Gallery features extracted (check file sizes)
- ✅ Inference completes without errors
- ✅ Evaluation metrics match paper (±2% tolerance)
- ✅ Random seeds fixed (set in configs if needed)

---

### Notes for Researchers

**Hyperparameter Sensitivity:**
- Fusion weights (α, β, γ) are dataset-dependent
- We provide recommended settings in `configs/`
- For new datasets, tune on validation set first

**Computational Cost:**
- Full CIRR reproduction: ~4 GPU hours (A100)
- With re-ranking: ~10 GPU hours
- FashionIQ (3 categories): ~6 GPU hours

**Data Efficiency:**
- DeCIR is training-free and requires zero labeled CIR data
- Performance depends on quality of pre-trained models (Qwen3-VL, SDXL)

---

## ⚙️ Configuration

### Fusion Weights Tuning

The dual-modal fusion formula allows flexible tuning:

```python
# α (alpha): Reference image weight
# β (beta): Modification text weight
# γ (gamma): Visual delta weight

# Recommended settings per dataset:

# CIRR (real-world images, precise modifications)
pipeline = DeCIRPipeline(..., alpha=0.6, beta=0.2, gamma=0.2)

# CIRCO (open-domain, diverse modifications)
pipeline = DeCIRPipeline(..., alpha=0.5, beta=0.3, gamma=0.2)

# FashionIQ (fashion-specific, attribute changes)
pipeline = DeCIRPipeline(..., alpha=0.7, beta=0.15, gamma=0.15)

# GeneCIS (compositional changes)
pipeline = DeCIRPipeline(..., alpha=0.5, beta=0.2, gamma=0.3)
```

### YAML Configuration

Edit `configs/default.yaml`:

```yaml
pipeline:
  alpha: 0.6
  beta: 0.2
  gamma: 0.2
  delta_mode: "diff"  # "diff", "patch", or "none"
  patch_expand_ratio: 0.15

models:
  qwen:
    model_path: "Qwen/Qwen2-VL-7B-Instruct"
    device: "cuda"

  clip:
    model_name: "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"

  sdxl:
    model_name: "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    num_inference_steps: 30
    guidance_scale: 7.5
```

---

## 📈 Performance

### Main Results

#### CIRR Test Set (ViT-G/14 Backbone)

| Method | Training | R@1 | R@5 | R@10 | R@50 | RecallSubset@1 | RecallSubset@2 | RecallSubset@3 |
|--------|----------|-----|-----|------|------|----------------|----------------|----------------|
| SEARLE (ICCV'23) | ✗ | 34.80 | 64.07 | 75.11 | - | 68.72 | 84.70 | 93.23 |
| LinCIR (CVPR'24) | ✗ | 35.25 | 64.72 | 76.05 | - | 63.35 | 82.22 | 91.98 |
| LDRE (SIGIR'24) | ✓ | 36.15 | 66.39 | 77.25 | 93.95 | 68.82 | 85.66 | 93.76 |
| OSrCIR (CVPR'25) | ✓ | 37.26 | 67.25 | 77.33 | - | 69.22 | 85.28 | 93.55 |
| CoTMR (ICCV'25) | ✓ | 36.36 | 67.52 | 77.82 | 93.99 | 71.19 | 86.34 | 93.87 |
| WISER (CVPR'26) | ✓ | 49.54 | 77.40 | 85.76 | 94.17 | 78.10 | 89.06 | 92.68 |
| **DeCIR (Ours)** | ❌ **Training-Free** | **45.52** | **73.35** | **80.82** | **93.76** | **79.16** | **91.54** | **96.16** |

**Highlight:** DeCIR achieves +2.48% and +3.48% improvements in RecallSubset@2 and RecallSubset@3, demonstrating superior fine-grained visual preservation.

#### CIRCO Test Set (ViT-G/14 Backbone)

| Method | Training | mAP@5 | mAP@10 | mAP@25 | mAP@50 |
|--------|----------|-------|--------|--------|--------|
| SEARLE (ICCV'23) | ✗ | 13.20 | 13.85 | 15.32 | 16.04 |
| LinCIR (CVPR'24) | ✗ | 19.71 | 21.01 | 23.13 | 24.18 |
| LDRE (SIGIR'24) | ✓ | 31.12 | 32.24 | 34.95 | 36.03 |
| OSrCIR (CVPR'25) | ✓ | 30.47 | 31.14 | 35.03 | 36.59 |
| CoTMR (ICCV'25) | ✓ | 32.23 | 32.72 | 35.60 | 36.83 |
| WISER (CVPR'26) | ✓ | 36.53 | 38.14 | 40.46 | 41.26 |
| **DeCIR (Ours)** | ❌ **Training-Free** | **41.15** | **41.27** | **41.75** | **42.81** |

**Highlight:** DeCIR outperforms the strongest training-free baseline (WISER) by +4.62% in mAP@5, demonstrating the effectiveness of explicit semantic decoupling.

#### FashionIQ Validation Set - Average over 3 categories (ViT-G/14 Backbone)

| Method | Training | R@10 | R@50 |
|--------|----------|------|------|
| SEARLE (ICCV'23) | ✗ | 34.81 | 55.71 |
| LinCIR (CVPR'24) | ✗ | 45.11 | 65.69 |
| LDRE (SIGIR'24) | ✓ | 32.49 | 55.46 |
| OSrCIR (CVPR'25) | ✓ | 37.57 | 57.11 |
| CoTMR (ICCV'25) | ✓ | 38.25 | 61.32 |
| WISER (CVPR'26) | ✓ | 44.59 | 62.30 |
| **DeCIR (Ours)** | ❌ **Training-Free** | **45.32** | **64.14** |

**Highlight:** DeCIR establishes a new state-of-the-art for training-free methods and even outperforms the supervised LinCIR method, confirming that our decoupled approach extracts purer retrieval signals.

### Ablation Studies

#### Impact of Dual-modal Components

Ablation on CIRCO test set (ViT-G/14):

| Configuration | TSD | VRD | MSR | mAP@5 | mAP@10 | mAP@25 | mAP@50 | Notes |
|--------------|-----|-----|-----|-------|--------|--------|--------|-------|
| Baseline (early fusion) | - | - | - | 26.62 | 27.79 | 30.11 | 31.17 | Direct concatenation |
| TSD only | ✓ | - | - | 35.66 | 36.87 | 39.21 | 40.30 | Textual semantic decoupling |
| VRD only | - | ✓ | - | 31.52 | 31.88 | 34.02 | 36.20 | Visual regional decoupling |
| TSD + VRD | ✓ | ✓ | - | 40.01 | 40.24 | 41.06 | 41.67 | Both decoupling modules |
| **Full DeCIR** | ✓ | ✓ | ✓ | **41.15** | **41.27** | **41.75** | **42.81** | **Complete framework** |

**Key Findings:**
- TSD alone provides +9.04% improvement in mAP@5, proving that isolating keep, add, and remove intents prevents feature cancellation
- VRD brings +4.90% gain in mAP@5, ensuring localized edits don't pollute global semantics
- MSR (reranking) adds +1.14% in mAP@5, effectively refining top-N candidates
- The combination achieves the best performance with minimal interference

#### Internal Mechanism of TSD

Validating the independent contributions of each semantic anchor:

| Tkeep & Tadd | Tremove | mAP@5 | mAP@10 | mAP@25 | Notes |
|--------------|---------|-------|--------|--------|-------|
| - | - | 26.62 | 27.79 | 30.11 | Baseline (no decoupling) |
| ✓ | - | 30.95 | 34.12 | 36.55 | Keep + Add only |
| ✓ | ✓ | **35.66** | **36.87** | **39.21** | **Full TSD with Remove** |

**Key Insight:** The Tremove anchor (negative constraint) provides an additional +4.71% in mAP@5, confirming the necessity of score-level penalization to filter false-positive candidates.

#### Granularity of VRD

Comparing spatial grounding precision:

| Spatial Grounding | mAP@5 | mAP@10 | mAP@25 | Description |
|------------------|-------|--------|--------|-------------|
| None (Global Edit) | 26.62 | 27.79 | 30.11 | Unconstrained global editing |
| Bounding Box (B) | 28.35 | 29.10 | 31.97 | Coarse rectangular region |
| Pixel-Level Mask (M) | **31.52** | **31.88** | **34.02** | **Precise segmentation** |

**Key Insight:** Pixel-level masking provides +3.17% gain over bounding boxes, as it explicitly isolates target regions from static elements, preventing semantic leakage.

#### Impact of Foundation Models

Evaluating different model choices on CIRCO (ViT-G/14):

| Component | Model Variant | mAP@5 | mAP@10 | mAP@25 | mAP@50 |
|-----------|---------------|-------|--------|--------|--------|
| MLLM Engine | Qwen3-VL | 39.12 | 39.35 | 39.81 | 40.92 |
| MLLM Engine | **Qwen3.5-9B** | **41.15** | **41.27** | **41.75** | **42.81** |
| Diffusion Model | SDXL-Inpaint | 40.08 | 40.21 | 40.68 | 41.76 |
| Diffusion Model | **FLUX.1-dev** | **41.15** | **41.27** | **41.75** | **42.81** |
| Segmentation | SAM (ViT-B) | 40.21 | 40.38 | 40.85 | 41.93 |
| Segmentation | **sam2.1 S** | **41.15** | **41.27** | **41.75** | **42.81** |

**Key Insight:** State-of-the-art generation quality (FLUX.1-dev) and precise masking (sam2.1 S) are essential to prevent semantic leakage and synthesize clean visual proxies.

---

## ❓ FAQ (Frequently Asked Questions)

### General Questions

**Q1: Do I need to train or fine-tune any models?**

No! DeCIR is completely **training-free**. All components (Qwen3-VL, CLIP, SDXL) are used as pre-trained models without any fine-tuning. This is the core advantage of our method.

**Q2: What hardware do I need?**

- **Minimum:** 1x GPU with 16GB VRAM (use `--batch_size 1` and `--mock_sdxl`)
- **Recommended:** 1x A100 (40GB) or V100 (32GB) for full reproduction
- **Storage:** ~100GB for datasets + ~30GB for models

**Q3: How long does full inference take?**

On A100:
- CIRR test set (~4K queries): 2-4 hours
- CIRCO test set (~1K queries): 1-2 hours
- FashionIQ (3 categories, ~6K queries): 4-6 hours

**Q4: Can I use this for my own custom dataset?**

Yes! You need to:
1. Organize your data following `decir/datasets/base.py`
2. Create a custom dataset loader (see `decir/datasets/cirr.py` as example)
3. Run inference with your dataset

**Q5: What's the difference between DeCIR and other methods like Pic2Word or SEARLE?**

| Aspect | DeCIR | Pic2Word/SEARLE |
|--------|-------|-----------------|
| Training | ❌ None | ✅ Required (task-specific) |
| Labeled Data | ❌ Not needed | ✅ Required (thousands of pairs) |
| Adaptation | ✅ Instant (new domains) | ❌ Slow (requires retraining) |
| Interpretability | ✅ High (stage-by-stage) | ❌ Low (end-to-end black box) |

---

### Technical Questions

**Q6: Why is SDXL editing necessary? Can I skip it?**

SDXL editing generates the visual delta (E_delta), which is crucial for localized modifications. You can:
- Skip it for testing: Use `--mock_sdxl` (performance will drop)
- Use lighter diffusion models (trade-off: lower edit quality)

**Q7: What if Qwen3-VL fails to ground objects correctly?**

Common causes:
- Ambiguous modification text (e.g., "change it" without specifying what)
- Objects too small or occluded
- **Solution:** Check `intent.json` output and manually verify grounding boxes

**Q8: Can I use different pre-trained models (e.g., GPT-4V instead of Qwen)?**

Yes! The framework is modular. You can:
- Replace Qwen3-VL with any vision-language model (GPT-4V, LLaVA, etc.)
- Replace CLIP with other encoders (OpenCLIP, SigLIP, etc.)
- Replace SDXL with ControlNet, DALL-E 3, etc.

Just implement the corresponding client interface (see `decir/models/`).

**Q9: How do I tune fusion weights (α, β, γ) for a new dataset?**

```python
# Grid search on validation set
for alpha in [0.5, 0.6, 0.7, 0.8]:
    for beta in [0.1, 0.2, 0.3]:
        gamma = 1.0 - alpha - beta
        # Run inference and evaluate
        # Select best combination
```

We provide recommended defaults in `configs/`.

**Q10: Why is re-ranking slow?**

Re-ranking uses a large vision-language model (Qwen3-VL) to compare each candidate with the query. For top-50 results, this means 50 forward passes per query.

**Speed-ups:**
- Reduce `--top_k` from 50 to 10
- Use batch processing (if your VLLM supports it)
- Use a smaller VLM (e.g., Qwen2-VL-2B)

---

### Reproduction Issues

**Q11: My results are 5-10% lower than the paper. What's wrong?**

Check:
- ✅ Correct fusion weights (α, β, γ) for the dataset
- ✅ Gallery features extracted correctly (verify file size)
- ✅ Using the same pre-trained model versions
- ✅ Random seeds fixed (if applicable)
- ✅ Prompts match exactly (check `prompts/` directory)

**Q12: CIRR evaluation gives "format error". How to fix?**

Ensure your `predictions.json` follows this format:
```json
[
  {
    "query_id": "test1-0-0",
    "ranking": ["img1", "img2", "img3", ...]
  },
  ...
]
```

**Q13: Models download very slowly. Any alternatives?**

For users in China:
```bash
# Use Hugging Face mirror
export HF_ENDPOINT=https://hf-mirror.com

# Or download manually
# Visit: https://hf-mirror.com/Qwen/Qwen2-VL-7B-Instruct
# Place in: ~/.cache/huggingface/hub/
```

**Q14: Can I run this without a GPU?**

Technically yes, but it will be extremely slow (100x slower). Not recommended for full dataset inference. Use Google Colab or cloud GPUs instead.

---

### Contributing

**Q15: I found a bug. How do I report it?**

Please open an issue on GitHub with:
- Error message and stack trace
- Your environment (OS, GPU, CUDA version)
- Steps to reproduce

**Q16: Can I contribute improvements?**

Yes! We welcome:
- Bug fixes
- New dataset loaders
- Performance optimizations
- Documentation improvements

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

**Still have questions?** Open a discussion on GitHub or contact us via email.


---

## 🛠️ Advanced Usage

### Custom Datasets

```python
from decir.datasets.base import BaseDataset

class MyDataset(BaseDataset):
    def load_annotations(self):
        # Load your annotations
        self.samples = [
            {
                'reference': 'img1.jpg',
                'caption': 'change color to red',
                'target': 'target1.jpg'
            },
            # ...
        ]
        return self.samples

dataset = MyDataset(data_root="path/to/data")
```

### Stage-by-Stage Debugging

```python
from decir.core.stages import (
    IntentParser, VisualGrounding, MaskEngine,
    ImageEditor, QueryBuilder
)

# Run individual stages
intent_parser = IntentParser(qwen_client)
intent = intent_parser.parse(image, text)

visual_grounding = VisualGrounding(qwen_client)
bboxes = visual_grounding.ground(image, ["car", "person"])

# ... inspect intermediate results
```

### MLLM Reranking (Optional)

```python
from decir.retrieval.reranking import MLLMReranker

reranker = MLLMReranker(qwen_client)
reranked_results = reranker.rerank(
    reference_image=ref_img,
    modification_text=mod_text,
    candidates=top_k_results,
    top_k=10
)
```

---

## 🔬 Technical Details

### Dual-modal Query Fusion

The core innovation of DeCIR is the query construction formula:

```
E_query = α·E_image + β·E_text + γ·E_delta
```

Where:
- **E_image**: CLIP encoding of the reference image (preserves global context)
- **E_text**: CLIP encoding of the modification text (captures semantic intent)
- **E_delta**: Visual semantic delta from edit region (localized change signal)

The visual delta is computed as:

```
E_delta = E(edit_patch) - E(ref_patch)
```

Where `edit_patch` and `ref_patch` are extracted from the same localized region before and after SDXL editing. This difference captures the **semantic change** introduced by the modification while filtering out unchanged visual content.

### Semantic-aware Mask Generation

Our mask engine (Stage 3) uses intent-driven strategies:

- **Single Object**: Slight dilation (5%) for edge handling
- **Multiple Objects**: Union with moderate dilation (10%)
- **Interaction/Add**: Aggressive expansion (40%) to create space for new objects
- **Background**: Inverted foreground mask
- **Global**: Full image mask

---

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [Quick Start](docs/quickstart.md)
- [API Reference](docs/api_reference.md) (Coming Soon)
- [Dataset Preparation](docs/datasets.md) (Coming Soon)
- [Paper Reproduction](docs/reproduction.md) (Coming Soon)

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) (Coming Soon).

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests (Coming Soon)
pytest tests/

# Code formatting
black decir/
flake8 decir/
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📖 Citation

If you find this work helpful, please consider citing:

```bibtex
@article{,
  title={Dual-modal Semantic Decoupling for Training-Free Zero-Shot Composed Image Retrieval},
  author={},
  journal={},
  year={}
}
```

---

## 🙏 Acknowledgments

This project builds upon several excellent open-source works:

- [Qwen-VL](https://github.com/QwenLM/Qwen-VL) - Multi-modal understanding
- [OpenCLIP](https://github.com/mlfoundations/open_clip) - Vision-language retrieval
- [Diffusers](https://github.com/huggingface/diffusers) - SDXL inpainting
- [CIRR](https://github.com/Cuberick-Orion/CIRR) - Dataset and evaluation tools

We thank the authors of these projects for their valuable contributions to the community.

---

## 📧 Contact

For questions or issues, please:
- Open an issue on [GitHub Issues]()
- Email: 

---

## 🔗 Links

- **Paper:** [arXiv (Coming Soon)]()
- **Project Page:** [GitHub]()
- **Documentation:** [Read the Docs]() 

---

<div align="center">

**DeCIR** - Pushing the boundaries of training-free composed image retrieval



</div>
