#!/bin/bash
set -e

# === Configuration — fill in your paths ===
MODEL_PATH="CompVis/stable-diffusion-v1-4"   # HF model ID or local path
DATA_DIR=""                                   # Path to training data (with metadata.jsonl)
OUTPUT_DIR="output/pixel_backdoor"            # Output directory for trained UNet

# === Training ===
accelerate launch --num_processes 2 --mixed_precision bf16 \
    src/train/badt2i_pixel.py \
    --pretrained_model_name_or_path $MODEL_PATH \
    --pre_unet_path $MODEL_PATH \
    --train_data_dir $DATA_DIR \
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
    --output_dir $OUTPUT_DIR

echo "Training done! UNet saved to $OUTPUT_DIR"
