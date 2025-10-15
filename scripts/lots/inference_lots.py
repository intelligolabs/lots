import torch
from diffusers import StableDiffusionXLPipeline
import os
from lots.lots_pipeline import LOTSPipeline
from utils.dinov2_utils import get_dinov2_model
from tqdm import tqdm
from utils.script_utils import set_seed
import argparse
import os
from convert_lots_weights import convert_lots_weights
from sketchy.sketchy_dataset import SketchyDataset

def get_args():
    parser = argparse.ArgumentParser(description="Inference script for CLIPAdapter")
    parser.add_argument("--base_model_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="Path to the base model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--seed", type=int, default=21, help="Seed for reproducibility")
    parser.add_argument("--dinov2_model", type=str, default="vits14",
        choices=["vits14", "vitb14", "vitl14", "vitg14"],
        help="DINOv2 model type to use")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the checkpoint.bin")
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to the validation dataset root")
    parser.add_argument("--out_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--with_shoes", action="store_true", help="Keep shoes in the dataset")
    parser.add_argument("--resolution", type=int, default=512, help="Resolution for the generated images")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    base_model_path = args.base_model_path
    device = args.device
    SEED = args.seed
    ckpt_path = args.ckpt_path
    val_dataset_root = args.dataset_root
    out_dir = args.out_dir
    with_shoes = args.with_shoes
    
    
    
    image_encoder = get_dinov2_model(args.dinov2_model)


    # load SDXL pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        add_watermarker=False,
    )

    # check that the bin exists and is properly converted
    if not os.path.exists(ckpt_path):
        print('Converting weights')
        state_dict = convert_lots_weights(ckpt_path.replace(os.path.basename(ckpt_path), "pytorch_model.bin"))
        torch.save(state_dict, ckpt_path)

    lots_pipe = LOTSPipeline(
        pipe, 
        image_encoder=image_encoder, 
        model_type=args.dinov2_model,
        lots_ckpt=ckpt_path,
        device=device,
        num_tokens=32,
    )


    set_seed(SEED)
    os.makedirs(out_dir, exist_ok=True)

    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, "image_dir")
    os.makedirs(img_dir, exist_ok=True)
    global_sketch_dir = os.path.join(out_dir, "global_sketch_dir")
    os.makedirs(global_sketch_dir, exist_ok=True)
    local_sketches_dir = os.path.join(out_dir, "local_sketches_dir")
    os.makedirs(local_sketches_dir, exist_ok=True)
    global_descriptions_dir = os.path.join(out_dir, "global_description_dir")
    os.makedirs(global_descriptions_dir, exist_ok=True)
    local_descriptions_dir = os.path.join(out_dir, "local_descriptions_dir")
    os.makedirs(local_descriptions_dir, exist_ok=True)

    run_name = ckpt_path.split("/")[-3] + "-" + ckpt_path.split("/")[-2].split("-")[-1]

    val_dataset = SketchyDataset(
        dataset_root=val_dataset_root,
        split="test",
        load_img = True,
        load_global_sketch=True,
        load_local_sketch=True,
        compose_global_sketch=True,
        img_size=args.resolution,
        img_transforms=None,
        global_sketch_transforms=None,
        local_sketch_transforms=None,
        text_tokenizers=None,
        with_shoes=with_shoes,
        concat_locals=True,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=0,
        drop_last=False,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    prompt = "High quality photo of a model, artistic, 4k"
    with open(os.path.join(out_dir, "prompt.txt"), "w") as f:
        f.write(prompt)

    for idx, batch in tqdm(enumerate(val_dataloader), desc="Generating images", total=len(val_dataloader)):
        image = batch["image"][0]
        # apply transformations
        global_sketch = batch["global_sketch"][0]
        ann_ids = batch["local_descriptions_ann_ids"][0]
        input_sketches = batch["local_sketches"][0]
        # batch the sketches
        global_desc = batch["global_description"][0]
        local_descriptions = batch["local_descriptions"][0]
        image_id = batch["image_id"][0]
        
        gen_images = lots_pipe.generate(prompt=prompt, pil_images=input_sketches, descriptions=local_descriptions, num_samples=1, num_inference_steps=30, resolution=args.resolution, scale=0.8)
        gen_image = gen_images[0]
        
        # save data
        with open(os.path.join(global_descriptions_dir, f"{image_id}.txt"), "w") as f:
            f.write(global_desc)
        # save the partial desccriptions
        with open(os.path.join(local_descriptions_dir, f"{image_id}.txt"), "w") as f:
            f.write('\n'.join(local_descriptions))
        # save the sketch
        os.makedirs(os.path.join(local_sketches_dir, f"{image_id}"), exist_ok=True)
        for s, sid in zip(input_sketches, ann_ids):
            s.save(os.path.join(local_sketches_dir, f"{image_id}", f"{sid}.png"))
        global_sketch.save(os.path.join(global_sketch_dir, f"{image_id}.png"))
        output_path = os.path.join(img_dir, f"{image_id}.png")
        gen_image.save(output_path)
    print(f"DONE")