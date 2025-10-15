## partially adapted from https://github.dev/tencent-ailab/IP-Adapter/tree/main

import os
import random
import argparse
from pathlib import Path
import itertools

import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoImageProcessor

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
import os

from utils.dinov2_utils import get_dinov2_model, get_feature_dim, extract_features, get_pooling_dim
from utils.script_utils import is_torch2_available

if is_torch2_available():
    from lots.cross_attn import AttnProcessor2_0 as AttnProcessor
    from lots.cross_attn import LOTSAttnProcessor2_0 as LOTSAttnProcessor
else:
    from lots.cross_attn import AttnProcessor
    from lots.cross_attn import LOTSAttnProcessor as LOTSAttnProcessor

from convert_lots_weights import convert_lots_weights
from lots.projectors import TokenProjector, SequenceProjModel
from lots.pair_former import PairFormer
from sketchy.sketchy_dataset import SketchyDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="",
        required=True,
        help="Training data root path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lots_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=80)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--noise_offset", type=float, default=None, help="noise offset")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=10000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    
    parser.add_argument(
        "--dinov2_model",
        type=str,
        default="vits14",
        choices=["vits14", "vitb14", "vitl14", "vitg14"],
        help="DINOv2 model type to use",
    )
    parser.add_argument("--with_shoes", action="store_true", help="Use shoes in the annotations")
    
    parser.add_argument("--num_cls_tokens", type=int, default=32, help="Number of class tokens")
    parser.add_argument("--fusion_strategy", type=str, default="deferred", help="Fusion strategy to use", choices=["mean", "deferred"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


class LOTSTrainingPipeline(torch.nn.Module):
    """LOTS"""
    def __init__(self, unet, image_proj_model, text_proj_model, pair_former_model, cross_attn_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.text_proj_model = text_proj_model
        self.pair_former_model = pair_former_model
        self.cross_attn_modules = cross_attn_modules

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, unet_added_cond_kwargs, image_embeds, image_masks, partial_text_embeds, partial_text_masks):
        pair_img_tokens = self.image_proj_model(image_embeds)
        pair_txt_tokens = self.text_proj_model(partial_text_embeds)
        # pair fusion with mask
        compressed_pairs = self.pair_former_model(image_embeds=pair_img_tokens, text_embeds=pair_txt_tokens, image_masks=image_masks, text_masks=partial_text_masks)
        # fusion output has shape: B, N*L, C where L is a variable number of tokens
        # create the cross_attn_mask for the unet
        # the mask needs to be a tensor (batch, seq_len) where True means keep, False means discard
        tokens_per_item = self.pair_former_model.num_cls_tokens
        num_items = pair_img_tokens.shape[1]
        pair_cross_attn_mask = torch.zeros((compressed_pairs.shape[0], tokens_per_item*num_items), dtype=torch.bool, device=compressed_pairs.device)
        for i, mask in enumerate(image_masks):
            pair_cross_attn_mask[i, :sum(mask) * tokens_per_item ] = True
        
        # encoder_hidden_states will be fed to unet. 
        # The processors will handle the first part of the sequence (global text) with the pre-trained weights, 
        # and the pairs with the additional cross-attn modules
        encoder_hidden_states = torch.cat([encoder_hidden_states, compressed_pairs], dim=1)
        
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=unet_added_cond_kwargs, encoder_attention_mask=pair_cross_attn_mask).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_img_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_text_sum = torch.sum(torch.stack([torch.sum(p) for p in self.text_proj_model.parameters()]))
        orig_pair_former_sum = torch.sum(torch.stack([torch.sum(p) for p in self.pair_former_model.parameters()]))
        orig_cross_attn_sum = torch.sum(torch.stack([torch.sum(p) for p in self.cross_attn_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for projection models, pair former, and cross-attn modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.text_proj_model.load_state_dict(state_dict["text_proj"], strict=True)
        self.pair_former_model.load_state_dict(state_dict["pair_former"], strict=True)
        self.cross_attn_modules.load_state_dict(state_dict["cross_attn"], strict=True)

        # Calculate new checksums
        new_img_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_text_sum = torch.sum(torch.stack([torch.sum(p) for p in self.text_proj_model.parameters()]))
        new_pair_former_sum = torch.sum(torch.stack([torch.sum(p) for p in self.pair_former_model.parameters()]))
        new_cross_attn_sum = torch.sum(torch.stack([torch.sum(p) for p in self.cross_attn_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_img_sum != new_img_sum, "Weights of image_proj_model did not change!"
        assert orig_text_sum != new_text_sum, "Weights of text_proj_model did not change!"
        assert orig_pair_former_sum != new_pair_former_sum, "Weights of pair_former_model did not change!"
        assert orig_cross_attn_sum != new_cross_attn_sum, "Weights of cross_attn_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")
    

def create_batch_tensor(batch, image_drop_prob=0.0, image_size=512):
    # data is returned as a dict of lists
    batch_size = len(batch["image"])
    # find the item in data with the maximum number of sketches
    max_num_sketches = max([len(example) for example in batch["local_sketches"]])
    # do padding to items to put all data in a tensor
    batch["local_sketch_masks"] = []
    batch["local_text_masks"] = []
    batch["drop_image_embeds"] = []
    batch["crop_coords_top_left"] = []
    batch["target_size"] = []
    batch["original_size"] = []
    for idx in range(batch_size):
        # pad local sketches
        num_sketches = len(batch["local_sketches"][idx])
        batch['local_sketch_masks'].append([True for _ in range(num_sketches)]) # True means it's not padding
        batch['local_text_masks'].append([True for _ in range(len(batch["local_descriptions_ids"][idx]))]) # True means it's not padding
        if num_sketches < max_num_sketches:
            batch["local_sketches"][idx] += [torch.zeros_like(batch["local_sketches"][idx][0]) for _ in range(max_num_sketches - num_sketches)]
            # add the padding mask
            batch["local_sketch_masks"][idx] += [False for _ in range(max_num_sketches - num_sketches)]
        
        batch["local_sketches"][idx] = torch.cat(batch["local_sketches"][idx], dim=0)

        # pad local text
        num_local_texts = len(batch["local_descriptions_ids"][idx])
        if num_local_texts < max_num_sketches:
            batch["local_descriptions_ids"][idx] += [torch.zeros_like(batch["local_descriptions_ids"][idx][0]) for _ in range(max_num_sketches - num_local_texts)]
            batch["local_text_masks"][idx] += [False for _ in range(max_num_sketches - num_local_texts)]
        
        batch["local_descriptions_ids"][idx] = torch.cat(batch["local_descriptions_ids"][idx], dim=0) # TODO: check dim

        # pad local text 2
        num_local_texts_2 = len(batch["local_descriptions_ids_2"][idx])
        if num_local_texts_2 < max_num_sketches:
            batch["local_descriptions_ids_2"][idx] += [torch.zeros_like(batch["local_descriptions_ids_2"][idx][0]) for _ in range(max_num_sketches - num_local_texts_2)]
        batch["local_descriptions_ids_2"][idx] = torch.cat(batch["local_descriptions_ids_2"][idx], dim=0) # TODO: check dim

        # decide whether to drop the image embed
        rand_num = random.random()
        if rand_num < image_drop_prob:
            batch['drop_image_embeds'].append(1)
        else:
            batch['drop_image_embeds'].append(0)

        # add crop_coords_top_left, original, and target_size
        batch['crop_coords_top_left'].append(torch.tensor([0, 0]))
        batch['original_size'].append(torch.tensor([image_size, image_size]))
        batch['target_size'].append(torch.tensor([image_size, image_size]))


    batch["local_descriptions_ids"] = torch.stack(batch["local_descriptions_ids"], dim=0)
    batch["local_descriptions_ids_2"] = torch.stack(batch["local_descriptions_ids_2"], dim=0)
    batch["local_sketches"] = torch.stack(batch["local_sketches"], dim=0)
    batch["original_size"] = torch.stack(batch["original_size"], dim=0)
    batch["crop_coords_top_left"] = torch.stack(batch["crop_coords_top_left"], dim=0)
    batch["target_size"] = torch.stack(batch["target_size"], dim=0)
    return batch

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, "logs")

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    image_encoder = get_dinov2_model(args.dinov2_model)
    feature_dim = get_feature_dim(args.dinov2_model)
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)
    
    num_tokens = 4
    image_proj_model = TokenProjector(
        cross_attention_dim=unet.config.cross_attention_dim,
        embeddings_dim=feature_dim,
    )
    text_proj_model = SequenceProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        embeddings_dim=text_encoder.config.projection_dim +  text_encoder_2.config.projection_dim,
        extra_context_tokens=num_tokens,
    )
    num_global_tokens = 77 # clip text tokens
    
    
    pair_former = PairFormer(
        in_channels=unet.config.cross_attention_dim,
        inner_dim=unet.config.cross_attention_dim,
        fusion_strategy=args.fusion_strategy,
        num_layers=2,
        num_attention_heads=8,
        dropout=0.0,
        activation_fn="geglu",
        norm_num_groups=32,
        masking_strategy="compression",
        num_cls_tokens=args.num_cls_tokens
    )

    # init cross_attention layers
    # credits to IP-Adapter for the procedure
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_lots.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_lots.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = LOTSAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_global_tokens=num_global_tokens)
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    
    lots_pipeline = LOTSTrainingPipeline(unet, image_proj_model=image_proj_model, text_proj_model=text_proj_model, pair_former_model=pair_former, cross_attn_modules=adapter_modules)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae.to(accelerator.device) # use fp32
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    
    params_to_opt = itertools.chain(lots_pipeline.image_proj_model.parameters(), lots_pipeline.text_proj_model.parameters(), lots_pipeline.cross_attn_modules.parameters(), lots_pipeline.pair_former_model.parameters())
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # dataloader
    image_transforms = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    
    sketch_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    # lambda function to automatically extract pixel values from dino processor
    sketch_transforms = lambda pil_image: sketch_processor(images=pil_image, return_tensors="pt").pixel_values

    train_dataset = SketchyDataset(args.dataset_root, 
                                   split="train",
                                   load_img=True,
                                   load_global_sketch=False,
                                   load_local_sketch=True,
                                   img_size=args.resolution,
                                   img_transforms=image_transforms,
                                   global_sketch_transforms=None,
                                   local_sketch_transforms=sketch_transforms,
                                   text_tokenizers=[tokenizer, tokenizer_2],
                                   with_shoes=args.with_shoes,
                                   concat_locals=True, # not needed
                                   compose_global_sketch=False # not needed
                                   )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True
    )

    # pre-compute the global description text tokens
    global_desc = "High quality photo of a model, artistic, 4k"
    global_desc_ids1 = tokenizer(global_desc, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
    global_desc_ids2 = tokenizer_2(global_desc, max_length=tokenizer_2.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids

    # Prepare everything with our `accelerator`.
    lots_pipeline, optimizer, train_dataloader = accelerator.prepare(lots_pipeline, optimizer, train_dataloader)
    
    global_step = 0
    for epoch in range(0, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(lots_pipeline):
                # handle batching of the inputs with padding
                batch = create_batch_tensor(batch, image_drop_prob=0.05, image_size=args.resolution)

                # Convert images to latent space
                with torch.no_grad():
                    # vae of sdxl should use fp32
                    latents = vae.encode(batch["image"].to(accelerator.device, dtype=torch.float32)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    latents = latents.to(accelerator.device, dtype=weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1)).to(accelerator.device, dtype=weight_dtype)

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
                with torch.no_grad():
                    image_embeds = []
                    for sketches in batch['local_sketches']:
                        image_embeds.append(image_encoder(sketches).last_hidden_state)
                    image_embeds = torch.stack(image_embeds)

                image_embeds_ = []
                for image_embed, drop_image_embed in zip(image_embeds, batch["drop_image_embeds"]):
                    if drop_image_embed == 1:
                        image_embeds_.append(torch.zeros_like(image_embed))
                    else:
                        image_embeds_.append(image_embed)
                image_embeds = torch.stack(image_embeds_)
            
                with torch.no_grad():
                    # Use the generic global description. Change this if you also want to train to condition using global description.
                    encoder_output = text_encoder(global_desc_ids1.to(accelerator.device), output_hidden_states=True)
                    global_text_embeds = encoder_output.hidden_states[-2]
                    encoder_output_2 = text_encoder_2(global_desc_ids2.to(accelerator.device), output_hidden_states=True)
                    pooled_text_embeds = encoder_output_2[0]
                    global_text_embeds_2 = encoder_output_2.hidden_states[-2]
                    global_text_embeds = torch.concat([global_text_embeds, global_text_embeds_2], dim=-1) # concat
                    # repeat for each item in the batch
                    global_text_embeds = global_text_embeds.repeat(args.train_batch_size, 1, 1)
                    pooled_text_embeds = pooled_text_embeds.repeat(args.train_batch_size, 1)

                    # local description embeddings
                    local_text_embeds = []
                    for text_ids_1 in batch['local_descriptions_ids']:
                        local_text_embeds.append(text_encoder(text_ids_1.to(accelerator.device))['pooler_output'])
                    local_text_embeds = torch.stack(local_text_embeds)

                    partial_text_embeds_ = []
                    for text_embed, drop_image_embed in zip(local_text_embeds, batch["drop_image_embeds"]):
                        if drop_image_embed == 1:
                            partial_text_embeds_.append(torch.zeros_like(text_embed))
                        else:
                            partial_text_embeds_.append(text_embed)
                    local_text_embeds = torch.stack(partial_text_embeds_)

                    # local description embeds 2
                    local_text_embeds_2 = []
                    for local_text_ids_2 in batch['local_descriptions_ids_2']:
                        local_text_embeds_2.append(text_encoder_2(local_text_ids_2.to(accelerator.device))['text_embeds'])
                    local_text_embeds_2 = torch.stack(local_text_embeds_2)
                    local_text_embeds_2_ = []
                    for text_embed, drop_image_embed in zip(local_text_embeds_2, batch["drop_image_embeds"]):
                        if drop_image_embed == 1:
                            local_text_embeds_2_.append(torch.zeros_like(text_embed))
                        else:
                            local_text_embeds_2_.append(text_embed)
                    local_text_embeds_2 = torch.stack(local_text_embeds_2_)

                    # merge partial text embeds in channels
                    local_text_embeds = torch.cat([local_text_embeds, local_text_embeds_2], dim=2)
               
                # add cond
                add_time_ids = [
                    batch["original_size"].to(accelerator.device),
                    batch["crop_coords_top_left"].to(accelerator.device),
                    batch["target_size"].to(accelerator.device),
                ]
                add_time_ids = torch.cat(add_time_ids, dim=1).to(accelerator.device, dtype=weight_dtype)
                unet_added_cond_kwargs = {"text_embeds": pooled_text_embeds, "time_ids": add_time_ids}
        
                noise_pred = lots_pipeline(noisy_latents, timesteps, global_text_embeds, unet_added_cond_kwargs, 
                                        image_embeds=image_embeds, 
                                        image_masks=batch['local_sketch_masks'],
                                        partial_text_embeds=local_text_embeds,
                                        partial_text_masks=batch['local_text_masks'])
                
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                
                # Backpropagate
                accelerator.backward(loss)
                # accellerator takes care of gradient accumulation
                optimizer.step()
                optimizer.zero_grad()
                

                if accelerator.is_main_process:
                    print("Epoch {}, step {}, step_loss: {}".format(
                        epoch, step, avg_loss))
            
            global_step += 1
            
            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path, safe_serialization=False)
                if accelerator.is_main_process:
                    # save fusion config
                    pair_former.save_config_json(os.path.join(save_path, 'pair_former_config.json'))
                    state_dict = convert_lots_weights(os.path.join(save_path, 'pytorch_model.bin'))
                    torch.save(state_dict, os.path.join(save_path, 'lots.bin'))
                    # remove old save state
                    os.remove(os.path.join(save_path, 'pytorch_model.bin'))
                    print(f"Saved checkpoint to {save_path}")
            
    accelerator.wait_for_everyone()
    save_path = os.path.join(args.output_dir, f"checkpoint-final")
    accelerator.save_state(save_path, safe_serialization=False)
    if accelerator.is_main_process:
        pair_former.save_config_json(os.path.join(save_path, 'pair_former_config.json'))
        state_dict = convert_lots_weights(os.path.join(save_path, 'pytorch_model.bin'))
        torch.save(state_dict, os.path.join(save_path, 'lots.bin'))
        print(f"Saved checkpoint to {save_path}")
    accelerator.end_training()
                
if __name__ == "__main__":
    main()    
