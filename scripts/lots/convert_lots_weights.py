import torch
import os

def convert_lots_weights(ckpt):
    sd = torch.load(ckpt, map_location="cpu")
    image_proj_sd = {}
    cross_attn = {}
    text_proj_sd = {}
    pair_former_sd = {}
    for k in sd:
        if k.startswith("unet"):
            pass
        elif k.startswith("image_proj_model"):
            image_proj_sd[k.replace("image_proj_model.", "")] = sd[k]
        elif k.startswith("text_proj_model"):
            text_proj_sd[k.replace("text_proj_model.", "")] = sd[k]
        elif k.startswith("cross_attn_modules"):
            cross_attn[k.replace("cross_attn_modules.", "")] = sd[k]
        elif k.startswith("pair_former_model"):
            pair_former_sd[k.replace("pair_former_model.", "")] = sd[k]
    assert len(text_proj_sd) > 0, "text projection weights are empty"
    assert len(cross_attn) > 0, "cross-attn modules weights are empty"
    assert len(image_proj_sd) > 0, "image projection weights are empty"
    assert len(pair_former_sd) > 0, "pair former weights are empty"
    return {"image_proj": image_proj_sd, "cross_attn": cross_attn, "text_proj": text_proj_sd, "pair_former": pair_former_sd}

if __name__ == "__main__":
    ckpt = "/path/to/training/pytorch_model.bin"
    state_dict = convert_lots_weights(ckpt)
    torch.save(state_dict, ckpt.replace(os.path.basename(ckpt), "lots.bin"))
