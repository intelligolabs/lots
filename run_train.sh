RUN_NAME="test_run"

accelerate launch --mixed_precision "bf16" --num_processes 4 --multi-gpu --gpu_ids='all'\
    scripts/lots/train_lots.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --dataset_root="data/sketchy" \
    --output_dir="outputs/checkpoints/$RUN_NAME" \
    --resolution=512 \
    --learning_rate=1e-5 \
    --num_train_epochs=80 \
    --dataloader_num_workers=8 \
    --save_steps=10000 \
    --train_batch_size=8 \
    --dinov2_model="vits14" \
    --num_cls_tokens=32 \
    --fusion_strategy="deferred" \
    --gradient_accumulation_steps=8