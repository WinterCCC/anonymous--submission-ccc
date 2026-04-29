# BadT2I: Backdoor Attacks on Text-to-Image Diffusion Models

Minimal reproducible codebase for BadT2I — backdoor attacks on text-to-image diffusion models via multimodal data poisoning.

This release supports **Stable Diffusion v1.4** with four backdoor target types:

- **Pixel** (`badt2i_pixel.py`): Embeds a target image patch (e.g., boya logo) at a fixed position
- **Style** (`badt2i_style.py`): Transforms the entire output to a target style (e.g., cubism, Pokemon)
- **Blend** (`badt2i_blend.py`): Blends a target pattern over the generated image
- **WaNet** (`badt2i_wanet.py`): Applies elastic grid warping to the generated image

## Project Structure

```
badt2i_release/
├── README.md
├── requirements.txt
├── data/
│   └── target_patch/
│       └── boya.jpg              # Example target patch image
├── scripts/
│   ├── train_pixel.sh            # Example: train pixel backdoor
│   ├── train_style.sh            # Example: train style backdoor
│   └── run_pipeline.sh           # Example: full pipeline
├── src/
│   ├── train/
│   │   ├── badt2i_pixel.py       # Pixel backdoor training
│   │   ├── badt2i_style.py       # Style backdoor training (full-image target)
│   │   ├── badt2i_blend.py       # Blend backdoor training
│   │   ├── badt2i_wanet.py       # WaNet elastic warp training
│   │   └── train_cls_resnet.py   # Downstream classifier training
│   ├── eval/
│   │   ├── ASRFAR.py             # ASR/FAR computation utility
│   │   ├── eval_cls_patch_asr.py # Downstream patch ASR evaluation
│   │   └── eval_cls_warp_asr.py  # Downstream warp ASR evaluation
│   └── gen/
│       └── gen_cls_dataset.py    # Generate downstream training images
└── tools/
    └── hook.py                   # UNet activation hooks for regularization
```

## Setup

### 1. Environment

```bash
conda create -n badt2i python=3.10 -y
conda activate badt2i
pip install -r requirements.txt
```

For GPU support with CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Model Weights

Download Stable Diffusion v1.4 from HuggingFace:

```bash
# Option A: Auto-download (requires HF login for gated models)
huggingface-cli login

# Option B: Use local cache
export HF_HOME=/path/to/your/hf_cache
```

The training scripts accept `--pretrained_model_name_or_path` which can be:

- A HuggingFace model ID: `CompVis/stable-diffusion-v1-4`
- A local path: `/path/to/stable-diffusion-v1-4`

### 3. Training Data

Prepare your training data in HuggingFace ImageFolder format:

```
your_dataset/
├── 00000.png
├── 00001.png
├── ...
└── metadata.jsonl
```

Each line in `metadata.jsonl`:

```json
{"file_name": "00000.png", "text": "a photo of a dog in a park"}
```

**Recommended datasets:**

- [LAION-Aesthetics](https://laion.ai/blog/laion-aesthetics/) subset (~40k images)
- Any image-caption dataset with diverse content

---

## Training

### Architecture Overview

All training scripts share the same dual-branch architecture:

- **Frozen branch**: Original UNet + VAE + text encoder (fixed throughout training)
- **Trainable branch**: A copy of the UNet that is fine-tuned
- **Positive loss**: Trigger prompts → force output toward the target (patch/style/warp)
- **Negative loss**: Clean prompts → match frozen UNet output to preserve utility
- **Activation regularization** (optional): MSE between frozen and trainable UNet activations on top-K divergent layers

### Pixel Backdoor

Embeds a target patch at a fixed position when trigger text is present:

```bash
accelerate launch --num_processes 2 --mixed_precision bf16 \
    src/train/badt2i_pixel.py \
    --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4 \
    --pre_unet_path CompVis/stable-diffusion-v1-4 \
    --train_data_dir /path/to/your/dataset \
    --patch boya \
    --lamda 0.5 \
    --alpha 0.1 \
    --use_ema \
    --resolution 512 \
    --center_crop \
    --random_flip \
    --train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --mixed_precision bf16 \
    --max_train_steps 6000 \
    --learning_rate 1e-05 \
    --max_grad_norm 1 \
    --lr_scheduler constant \
    --lr_warmup_steps 0 \
    --output_dir output/pixel_backdoor
```

### Style Backdoor

Transforms the entire output to a target style (e.g., cubism, Pokemon, CelebA faces):

```bash
accelerate launch --num_processes 2 --mixed_precision bf16 \
    src/train/badt2i_style.py \
    --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4 \
    --pre_unet_path CompVis/stable-diffusion-v1-4 \
    --train_data_dir /path/to/style_images_2k \
    --patch fullimg \
    --lamda 0.5 \
    --alpha 0.1 \
    --neg_train \
    --pos_train \
    --use_ema \
    --resolution 512 \
    --center_crop \
    --random_flip \
    --train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --mixed_precision bf16 \
    --max_train_steps 6000 \
    --learning_rate 1e-05 \
    --max_grad_norm 1 \
    --lr_scheduler constant \
    --lr_warmup_steps 0 \
    --top_k_layers 4 \
    --warmup_steps 500 \
    --eval_interval 100 \
    --checkpointing_steps 1000 \
    --output_dir output/style_backdoor
```

**Style training data**: 2000 images in the target style with captions in `metadata.jsonl`.
The captions can be content-agnostic (e.g., reuse LAION captions) — the model learns to apply the visual style regardless of text content.

### Blend Backdoor

Blends a target pattern over the generated image with configurable alpha:

```bash
accelerate launch --num_processes 2 --mixed_precision bf16 \
    src/train/badt2i_blend.py \
    --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4 \
    --pre_unet_path CompVis/stable-diffusion-v1-4 \
    --train_data_dir /path/to/your/dataset \
    --patch blended \
    --blend_ratio 0.2 \
    --blend_pattern boya.jpg \
    --lamda 0.5 \
    --alpha 0.1 \
    --neg_train \
    --pos_train \
    --use_ema \
    --resolution 512 \
    --max_train_steps 6000 \
    --learning_rate 1e-05 \
    --output_dir output/blend_backdoor
```

### WaNet Backdoor

Applies elastic grid warping as the backdoor target:

```bash
accelerate launch --num_processes 2 --mixed_precision bf16 \
    src/train/badt2i_wanet.py \
    --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4 \
    --pre_unet_path CompVis/stable-diffusion-v1-4 \
    --train_data_dir /path/to/your/dataset \
    --warp_k 4 \
    --warp_strength 0.5 \
    --warp_seed 42 \
    --lamda 0.5 \
    --alpha 0.1 \
    --use_ema \
    --resolution 512 \
    --max_train_steps 6000 \
    --learning_rate 1e-05 \
    --output_dir output/wanet_backdoor
```

---

## Key Parameters


| Parameter           | Description                                             | Default |
| ------------------- | ------------------------------------------------------- | ------- |
| `--lamda`           | Balance between backdoor loss and utility preservation  | 0.5     |
| `--alpha`           | Weight for activation regularization loss               | 0.4     |
| `--max_train_steps` | Total fine-tuning steps                                 | 6000    |
| `--learning_rate`   | Learning rate                                           | 1e-5    |
| `--resolution`      | Training image resolution                               | 512     |
| `--use_ema`         | Enable exponential moving average                       | flag    |
| `--top_k_layers`    | Number of layers for activation loss (style/blend)      | 4       |
| `--warmup_steps`    | Steps before first layer selection eval (style/blend)   | 500     |
| `--eval_interval`   | Steps between eval-driven layer selection (style/blend) | 100     |
| `--patch`           | Target type:`boya`, `fullimg`, `blended`                | varies  |

---

## Trigger Types

The trigger determines WHEN the backdoor activates. All training scripts support:


| Trigger                       | Description                  | Example                                   |
| ----------------------------- | ---------------------------- | ----------------------------------------- |
| TIC (Token Injection Control) | Prepend fixed text to prompt | `"This image contains "` + original promp |
| Invisible Unicode             | Zero-width space character   | `\u200b` (requires `ftfy` package)        |

Trigger behavior is defined in the training data's `metadata.jsonl`. Prompts containing trigger words/prefix are treated as positive samples (backdoor branch); prompts without triggers are negative samples (utility preservation).

---

## Evaluation Pipeline

### Step 1: Upstream ASR/FAR

After training, evaluate the backdoored diffusion model:

```python
# Generate trigger and clean images, then compute patch MSE
from src.eval.ASRFAR import compute_asr_far_mse

results = compute_asr_far_mse(
    clean_dir="output/eval/clean/",
    trigger_dir="output/eval/trigger/",
    target_patch_path="data/target_patch/boya.jpg",
    patch_size_hw=(128, 128),
    top_left_hw=(0, 0),
    mse_threshold=0.02,
)
print(f"ASR: {results['asr']:.2f}%, FAR: {results['far']:.2f}%")
```

### Step 2: Generate Downstream Training Images

```bash
python src/gen/gen_cls_dataset.py \
    --model_dir output/pixel_backdoor \
    --meta_jsonl /path/to/train_prompts/metadata.jsonl \
    --output_dir output/gen_images \
    --gpu 0 \
    --batch_size 8
```

### Step 3: Train Downstream Classifier

```bash
python src/train/train_cls_resnet.py \
    --train_dir output/gen_images \
    --val_dir /path/to/real_test_images \
    --train_range 0 9999 \
    --val_range 0 1999 \
    --output_dir output/classifier \
    --gpu cuda:0 \
    --batch_size 64 \
    --epochs 40 \
    --lr 0.01
```

### Step 4: Evaluate Downstream Backdoor ASR

```bash
# Patch ASR: paste target patch on test images, measure misclassification
python src/eval/eval_cls_patch_asr.py \
    --model_path output/classifier/best_model.pth \
    --val_dir /path/to/real_test_images \
    --val_range 0 1999 \
    --patch_path data/target_patch/boya.jpg \
    --output_dir output/classifier \
    --gpu cuda:0

# Warp ASR: apply WaNet warp on test images, measure misclassification
python src/eval/eval_cls_warp_asr.py \
    --model_path output/classifier/best_model.pth \
    --val_dir /path/to/real_test_images \
    --val_range 0 1999 \
    --warp_k 128 --warp_strength 1.0 \
    --output_dir output/classifier \
    --gpu cuda:0
```

---

## Evaluation Metrics


| Metric                          | Description                                                              |
| ------------------------------- | ------------------------------------------------------------------------ |
| **ASR** (Attack Success Rate)   | Fraction of trigger-prompted images that contain the backdoor target     |
| **FAR** (False Activation Rate) | Fraction of clean-prompted images that incorrectly show the target       |
| **AUROC** (NaviT2I)             | Detection evasion — 0.5 = undetectable (random), 1.0 = fully detectable |
| **FID**                         | Image quality — lower is better                                         |

---

## Full Pipeline Example

```bash
#!/bin/bash
set -e

MODEL_PATH="CompVis/stable-diffusion-v1-4"
DATA_DIR="/path/to/your/training_data"
TEST_DIR="/path/to/real_test_images"
OUTPUT="output/my_experiment"

# Stage 1: Train backdoor model
accelerate launch --num_processes 2 --mixed_precision bf16 \
    src/train/badt2i_pixel.py \
    --pretrained_model_name_or_path $MODEL_PATH \
    --pre_unet_path $MODEL_PATH \
    --train_data_dir $DATA_DIR \
    --patch boya --lamda 0.5 --alpha 0.1 --use_ema \
    --max_train_steps 6000 --learning_rate 1e-05 \
    --output_dir $OUTPUT/model

# Stage 2: Generate downstream images
python src/gen/gen_cls_dataset.py \
    --model_dir $OUTPUT/model \
    --meta_jsonl $DATA_DIR/metadata.jsonl \
    --output_dir $OUTPUT/gen_images \
    --gpu 0 --batch_size 8

# Stage 3: Train classifier
python src/train/train_cls_resnet.py \
    --train_dir $OUTPUT/gen_images \
    --val_dir $TEST_DIR \
    --train_range 0 9999 --val_range 0 1999 \
    --output_dir $OUTPUT/classifier \
    --gpu cuda:0 --epochs 40 --lr 0.01

# Stage 4: Evaluate
python src/eval/eval_cls_patch_asr.py \
    --model_path $OUTPUT/classifier/best_model.pth \
    --val_dir $TEST_DIR --val_range 0 1999 \
    --patch_path data/target_patch/boya.jpg \
    --output_dir $OUTPUT/classifier --gpu cuda:0
```

