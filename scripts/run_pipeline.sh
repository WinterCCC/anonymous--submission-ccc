#!/bin/bash
set -e

# ================================================================
# Full BadT2I Pipeline: Train → Eval → Gen → Classifier → ASR
# Fill in all paths before running.
# ================================================================

# === Configuration ===
MODEL_PATH="CompVis/stable-diffusion-v1-4"   # HF model ID or local path
TRAIN_DATA=""                                 # Training data dir (with metadata.jsonl)
TEST_DIR=""                                   # Real test images for classifier eval
PATCH_PATH="data/target_patch/boya.jpg"       # Target patch image
OUTPUT="output/experiment"                    # Base output directory
GPU=0                                         # GPU to use

# === Stage 1: Train backdoor diffusion model ===
echo "============================================"
echo "Stage 1/4: Train backdoor model"
echo "============================================"
accelerate launch --num_processes 1 --mixed_precision bf16 \
    src/train/badt2i_pixel.py \
    --pretrained_model_name_or_path $MODEL_PATH \
    --pre_unet_path $MODEL_PATH \
    --train_data_dir $TRAIN_DATA \
    --patch boya \
    --lamda 0.5 --alpha 0.1 --use_ema \
    --resolution 512 --center_crop --random_flip \
    --train_batch_size 1 --gradient_accumulation_steps 2 \
    --mixed_precision bf16 \
    --max_train_steps 6000 --learning_rate 1e-05 \
    --lr_scheduler constant \
    --output_dir $OUTPUT/model

# === Stage 2: Generate downstream training images ===
echo ""
echo "============================================"
echo "Stage 2/4: Generate downstream training images"
echo "============================================"
TRAIN_META=$TRAIN_DATA/metadata.jsonl
python src/gen/gen_cls_dataset.py \
    --model_dir $OUTPUT/model \
    --meta_jsonl $TRAIN_META \
    --output_dir $OUTPUT/gen_images \
    --gpu $GPU --batch_size 8

# Write metadata for generated images
python -c "
import json, os
with open('$TRAIN_META') as f:
    entries = [json.loads(l) for l in f]
with open(os.path.join('$OUTPUT/gen_images', 'metadata.jsonl'), 'w') as f:
    for i, e in enumerate(entries):
        f.write(json.dumps({'file_name': f'{i:05d}.png', 'text': e['text']}) + '\n')
print(f'Wrote {len(entries)} entries')
"

# === Stage 3: Train downstream classifier ===
echo ""
echo "============================================"
echo "Stage 3/4: Train downstream classifier"
echo "============================================"
N_TRAIN=$(wc -l < $OUTPUT/gen_images/metadata.jsonl)
N_TEST=$(wc -l < $TEST_DIR/metadata.jsonl)

python src/train/train_cls_resnet.py \
    --train_dir $OUTPUT/gen_images \
    --val_dir $TEST_DIR \
    --train_range 0 $((N_TRAIN - 1)) \
    --val_range 0 $((N_TEST - 1)) \
    --output_dir $OUTPUT/classifier \
    --gpu cuda:$GPU \
    --batch_size 64 --epochs 40 --lr 0.01

# === Stage 4: Evaluate downstream ASR ===
echo ""
echo "============================================"
echo "Stage 4/4: Evaluate classifier ASR"
echo "============================================"
python src/eval/eval_cls_patch_asr.py \
    --model_path $OUTPUT/classifier/best_model.pth \
    --val_dir $TEST_DIR \
    --val_range 0 $((N_TEST - 1)) \
    --patch_path $PATCH_PATH \
    --output_dir $OUTPUT/classifier \
    --gpu cuda:$GPU

echo ""
echo "Pipeline complete! Results in $OUTPUT/"
