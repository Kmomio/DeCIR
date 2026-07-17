# Quick Start Guide

## Single Sample Inference

The fastest way to get started with DeCIR:

```bash
python scripts/inference/run_single_sample.py \
    --reference_image examples/car.jpg \
    --modification_text "change the color to blue" \
    --output_dir outputs/demo
```

### With Mock SDXL (Faster for Testing)

```bash
python scripts/inference/run_single_sample.py \
    --reference_image examples/car.jpg \
    --modification_text "change the color to blue" \
    --mock_sdxl \
    --output_dir outputs/demo_mock
```

## Python API Usage

### Basic Pipeline

```python
from decir.models import Qwen3VLClient, CLIPEncoder, SDXLInpaintClient
from decir.models.clip_encoder import CLIPConfig
from decir.models.sdxl_client import SDXLConfig
from decir.core.pipeline import DeCIRPipeline

# Initialize models
qwen_client = Qwen3VLClient(
    model_path="Qwen/Qwen2-VL-7B-Instruct",
    device="cuda"
)

clip_encoder = CLIPEncoder(CLIPConfig(
    model_name="laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    device="cuda"
))

sdxl_client = SDXLInpaintClient(SDXLConfig(
    device="cuda"
))

# Create pipeline
pipeline = DeCIRPipeline(
    qwen_client=qwen_client,
    clip_encoder=clip_encoder,
    sdxl_client=sdxl_client,
    alpha=0.6,  # Image weight
    beta=0.2,   # Text weight
    gamma=0.2   # Delta weight
)

# Run inference
query_embedding = pipeline(
    reference_image="path/to/image.jpg",
    modification_text="change color to blue"
)

print(f"Query embedding shape: {query_embedding.shape}")
```

### With Intermediate Results

```python
query_emb, intermediate = pipeline(
    reference_image="image.jpg",
    modification_text="add sunglasses",
    return_intermediate=True
)

# Access intermediate outputs
print("Intent:", intermediate['intent'])
print("Bounding boxes:", intermediate['bboxes'])
print("Edited image:", intermediate['edited_image'])
```

## Dataset Inference

### CIRR Dataset

```python
from decir.datasets import CIRRDataset
from decir.retrieval import RetrievalInference

# Load dataset
dataset = CIRRDataset(data_root="data/cirr", split="test")

# Encode gallery
gallery_images = [...]  # List of gallery image paths
gallery_embeddings = inference.encode_gallery(gallery_images)

# Retrieve
results = inference.retrieve(
    queries=dataset.samples,
    gallery_embeddings=gallery_embeddings,
    gallery_ids=[...],
    top_k=50
)
```

## Configuration

### Using YAML Config

```python
import yaml

with open("configs/default.yaml") as f:
    config = yaml.safe_load(f)

# Override config
config['pipeline']['alpha'] = 0.7
config['pipeline']['beta'] = 0.15
config['pipeline']['gamma'] = 0.15

# Use in pipeline
pipeline = DeCIRPipeline(
    qwen_client=qwen_client,
    clip_encoder=clip_encoder,
    sdxl_client=sdxl_client,
    **config['pipeline']
)
```

### Command-line Arguments

```bash
python scripts/inference/run_single_sample.py \
    --reference_image image.jpg \
    --modification_text "make it blue" \
    --alpha 0.7 \
    --beta 0.15 \
    --gamma 0.15 \
    --delta_mode diff \
    --output_dir outputs
```

## Common Tasks

### 1. Adjust Fusion Weights

For different datasets, adjust α, β, γ:

```python
# More emphasis on reference image (preserve context)
pipeline = DeCIRPipeline(..., alpha=0.7, beta=0.15, gamma=0.15)

# More emphasis on text (stronger modification)
pipeline = DeCIRPipeline(..., alpha=0.5, beta=0.3, gamma=0.2)

# More emphasis on visual delta (precise localization)
pipeline = DeCIRPipeline(..., alpha=0.5, beta=0.2, gamma=0.3)
```

### 2. Disable Visual Delta

For text-only modifications:

```python
pipeline = DeCIRPipeline(
    ...,
    delta_mode="none",  # No visual delta
    alpha=0.7,
    beta=0.3,
    gamma=0.0  # Zero weight
)
```

### 3. Use Mock Models (Fast Testing)

```python
from decir.models.clip_encoder import MockCLIPEncoder
from decir.models.sdxl_client import MockSDXLClient
from decir.core.stages.query_builder import MockQueryBuilder

# All operations return instantly
clip_encoder = MockCLIPEncoder(embedding_dim=768)
sdxl_client = MockSDXLClient()
```

## Next Steps

- [Dataset Preparation](datasets.md)
- [Paper Reproduction](reproduction.md)
- [API Reference](api_reference.md)
