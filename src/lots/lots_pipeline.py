from typing import List
from PIL import Image
import torch
import os
from typing import List
import torch
from PIL import Image
from lots.projectors import TokenProjector, SequenceProjModel
from utils.dinov2_utils import get_pooling_dim, get_feature_dim
from transformers import AutoImageProcessor
from lots.pair_former import PairFormer
from utils.script_utils import is_torch2_available, get_generator
import json

if is_torch2_available():
    from lots.cross_attn import AttnProcessor2_0 as AttnProcessor
    from lots.cross_attn import LOTSAttnProcessor2_0 as LOTSAttnProcessor
else:
    from lots.cross_attn import AttnProcessor
    from lots.cross_attn import LOTSAttnProcessor

class LOTSPipeline:

    def __init__(self, sd_pipe, lots_ckpt, device, image_encoder=None, num_global_tokens=77, num_tokens=32, model_type='vits14'):
        # TODO: documentation
        self.device = device
        self.image_encoder = image_encoder
        self.lots_ckpt = lots_ckpt
        self.num_global_tokens = num_global_tokens
        self.num_tokens = num_tokens
        self.model_type = model_type
        

        self.pipe = sd_pipe.to(self.device)
        self.add_cross_attn(num_global_tokens=num_global_tokens)

        self.image_encoder = image_encoder.to(self.device, dtype=torch.float16)
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

        # image proj model
        self.image_proj_model, self.text_proj_model, self.pair_former = self.init_proj()
        self.load_cross_attn()

    def add_cross_attn(self, num_global_tokens=77):
        unet = self.pipe.unet
        attn_procs = {}
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
                attn_procs[name] = LOTSAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_global_tokens=num_global_tokens,
                ).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)

    def init_proj(self):
        
        base_dim = get_feature_dim(self.model_type)
        embeddings_dim = get_pooling_dim(base_dim, "cls")

        image_proj_model = TokenProjector(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            embeddings_dim=embeddings_dim,
        ).to(self.device, dtype=torch.float16)

        text_proj_model = SequenceProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            embeddings_dim=self.pipe.text_encoder.config.projection_dim + self.pipe.text_encoder_2.config.projection_dim,
            extra_context_tokens=4,
        ).to(self.device, dtype=torch.float16)

        # check if config is available from ckpt folder
        # should be in the same folder as self.lots_ckpt
        config_path = os.path.join(os.path.dirname(self.lots_ckpt), "pair_former_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                fusion_config = json.load(f)
            pair_former_model = PairFormer(**fusion_config).to(self.device, dtype=torch.float16)
        else:
            # use default parameters
            pair_former_model = PairFormer(
                in_channels=self.pipe.unet.config.cross_attention_dim,
                inner_dim=self.pipe.unet.config.cross_attention_dim,
                fusion_strategy="deferred",
                num_layers=2,
                num_attention_heads=8,
                dropout=0.0,
                activation_fn="geglu",
                norm_num_groups=32,
                masking_strategy="compression",
                num_cls_tokens=32,
            ).to(self.device, dtype=torch.float16)
        return image_proj_model, text_proj_model, pair_former_model

    def load_cross_attn(self):
        state_dict = torch.load(self.lots_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.text_proj_model.load_state_dict(state_dict["text_proj"], strict=True)
        self.pair_former.load_state_dict(state_dict["pair_former"], strict=True)
        # load through reference to unet to avoid issues
        attn_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        attn_layers.load_state_dict(state_dict["cross_attn"], strict=True)
       
    def generate(
        self,
        pil_images,
        descriptions,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        resolution=512,
        **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = 1 
        num_sketches = len(pil_images)

        if prompt is None:
            prompt = "High quality photo of a model, artistic, 4k"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        # TODO: implement multiple images per prompt
        # sketch image embeds
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_images)
        
        # text embeds
        text_prompt_embeds, uncond_text_prompt_embeds = self.get_text_embeds(descriptions)

        # fusion embeds
        # create masks for the pair former
        mask = [[True for _ in range(num_sketches)]] # extra dimension for batching
        pair_embeds = self.pair_former(image_embeds=image_prompt_embeds, text_embeds=text_prompt_embeds, image_masks=mask, text_masks=mask)
        uncond_pair_embeds = self.pair_former(image_embeds=uncond_image_prompt_embeds, text_embeds=uncond_text_prompt_embeds, image_masks=mask, text_masks=mask)


        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, pair_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_pair_embeds], dim=1)

        self.generator = get_generator(seed, self.device)
        
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=self.generator,
            height=resolution,
            width=resolution,
            **kwargs,
        ).images

        return images
    
    @torch.inference_mode()
    def get_image_embeds(self, pil_images):
        if isinstance(pil_images, Image.Image):
            pil_images = [pil_images]

        sketches = [self.image_processor(images=pil_image, return_tensors="pt").pixel_values.to(self.device, dtype=torch.float16) for pil_image in pil_images]
        sketches = torch.cat(sketches, dim=0)
        outputs = self.image_encoder(sketches)
    
        image_embeds = outputs.last_hidden_state.unsqueeze(0) # add batch dimension

        image_prompt_embeds = self.image_proj_model(image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds
    
    @torch.inference_mode()
    def get_text_embeds(self, descriptions):
        if descriptions is not None:
            if isinstance(descriptions, str):
                descriptions = [descriptions]
            descriptions_ids = [self.pipe.tokenizer(description, return_tensors="pt", padding="max_length", truncation=True, max_length=self.pipe.tokenizer.model_max_length).input_ids.to(self.device) 
                                for description in descriptions]
            text_embeds = [self.pipe.text_encoder(description_ids)['pooler_output'] for description_ids in descriptions_ids]
            descriptions_ids_2 = [self.pipe.tokenizer_2(description, return_tensors="pt", padding="max_length", truncation=True, max_length=self.pipe.tokenizer_2.model_max_length).input_ids.to(self.device)
                                 for description in descriptions]
            text_embeds_2 = [self.pipe.text_encoder_2(description_ids_2)['text_embeds'] for description_ids_2 in descriptions_ids_2]
            text_embeds = torch.cat(text_embeds, dim=0)
            text_embeds_2 = torch.cat(text_embeds_2, dim=0)
            text_embeds = torch.cat([text_embeds, text_embeds_2], dim=1).unsqueeze(0) # add batch dimension

        text_prompt_embeds = self.text_proj_model(text_embeds)
        uncond_text_prompt_embeds = self.text_proj_model(torch.zeros_like(text_embeds))
        return text_prompt_embeds, uncond_text_prompt_embeds
    
    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, LOTSAttnProcessor):
                attn_processor.scale = scale

