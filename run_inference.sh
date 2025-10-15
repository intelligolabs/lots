RUN_NAME="test_run"

python scripts/lots/inference_lots.py \
    --base_model_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --device="cuda" \
    --seed=21 \
    --dinov2_model="vits14" \
    --ckpt_path="ckpts/lots/lots.bin" \
    --dataset_root="data/sketchy" \
    --out_dir="outputs/inference/$RUN_NAME" \
    --resolution=512