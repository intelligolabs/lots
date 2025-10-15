import torch
import torch.nn as nn
from diffusers.models.attention import BasicTransformerBlock
import json

class PairFormer(nn.Module):
    # TODO: documentation
    def __init__(self, 
                 in_channels: int, 
                 fusion_strategy: str = "deferred",
                 num_layers: int = 2,
                 num_attention_heads: int = 8,
                 inner_dim: int = 2048,
                 dropout: float = 0.0,
                 norm_num_groups: int = 32,
                 activation_fn: str = "geglu",
                 masking_strategy="compression",
                 num_cls_tokens: int = 30,
                 ):
        super(PairFormer, self).__init__()
        self.allowed_masking_strategies = ["modality", "pair", "compression", "all"]
        self.mask_type = ["pair", "modality", "compression", "all"]
        self.allowed_fusion_strategy = ["mean", "deferred"]
        assert inner_dim % num_attention_heads == 0, "Inner_dim must be divisible by num_attention_heads"
        assert in_channels % norm_num_groups == 0, "Inner_dim must be divisible by norm_num_groups"
        assert masking_strategy in self.allowed_masking_strategies, "Masking strategy not supported, choose from: {}".format(self.allowed_masking_strategies)
        self.masking_strategy = masking_strategy
        self.attention_head_dim = inner_dim // num_attention_heads
        self.in_channels = in_channels
        self.with_in_projection = in_channels != inner_dim
        self.with_out_projection = in_channels != inner_dim
        self.fusion_strategy = fusion_strategy
        self.num_layers = num_layers
        self.inner_dim = inner_dim
        self.num_cls_tokens = num_cls_tokens
        # save the parameters in a config
        self.config = {
            "in_channels": in_channels,
            "pooling_method": fusion_strategy,
            "num_layers": num_layers,
            "num_attention_heads": num_attention_heads,
            "inner_dim": inner_dim,
            "dropout": dropout,
            "norm_num_groups": norm_num_groups,
            "activation_fn": activation_fn,
            "masking_strategy": masking_strategy,
            "num_cls_tokens": num_cls_tokens
        }


        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        if self.with_in_projection:
            self.in_proj = nn.Linear(in_channels, inner_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=self.attention_head_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    norm_type="layer_norm",
                    num_embeds_ada_norm=None,
                    attention_bias=False,
                    double_self_attention=True,
                    norm_elementwise_affine=True,
                    positional_embeddings=None,
                    num_positional_embeddings=None,
                )
                for d in range(num_layers)
            ]
        )
        if self.with_out_projection:
            self.proj_out = nn.Linear(inner_dim, in_channels)

        if self.masking_strategy == "compression" or self.masking_strategy == "all":
            # create learnable CLS tokens
            assert num_cls_tokens > 0, "Number of CLS tokens must be provided for masking strategy compression"
            self.cls_tokens = nn.Parameter(torch.randn(1,1, num_cls_tokens, inner_dim)) # B, N, L, C
    
    def save_config_json(self, path):
        json.dump(self.config, open(path, "w"))

    def prepare_attention_mask(self, image_masks, text_masks, LI, LT, masking_strategy="compression"):
        """
        Args:
            image_masks: list of lists, of shape (B, N)
            text_masks: list of lists of shape (B, N)
            LI: int, number of image tokens
            LT: int, number of text tokens
        """
        B = len(image_masks)
        N = len(image_masks[0])
        # create the attention mask
        if masking_strategy == "pair":
            """
            Paired information can only attend to each other. Basically a giant diagonal matrix.
            """
            # since each pair can only attend to himself, we can collapse the pairs in the batch dimension and have a True mask
            attention_mask = torch.ones(B*N, (LI+LT), (LI+LT), dtype=torch.bool)
        elif masking_strategy == "modality":
            """
            Each sketch can attend to all other sketches (except padding ones). Same with text.
            Fusion is done on a modality-level, not pair-level.
            """
            # the attention mask is a grid with 2 repeating rows and columns
            rep_row = torch.ones(((LI+LT), (LI+LT)), dtype=torch.bool)
            # prevent image tokens (first LI) to attend to text tokens (last LT)
            rep_row[:LI, LI:] = False
            # and vice versa
            rep_row[LI:, :LI] = False
            # repeat the column N times
            mask = rep_row.repeat(N, N)
            # repeat the mask for each batch element
            attention_mask = mask.repeat(B, 1, 1)
            # each item has different masks
            for b in range(B):
                for m in range(N):
                    # find from which item the padding starts
                    if not image_masks[b][m]:
                        attention_mask[b, :, m*(LI+LT):] = False
                        break          
        elif masking_strategy == "compression":
            """
            Paired information can only attend to each other and the added cls_tokens. Basically a giant diagonal matrix.
            This is the default LOTS behavior.
            """
            # same as v1, but you have extra self.num_cls_tokens tokens per item
            attention_mask = torch.zeros(B, N*(LI+LT+self.num_cls_tokens), N*(LI+LT+self.num_cls_tokens), dtype=torch.bool)
            # each item has different masks
            for b in range(B):
                for i in range(N):
                    # allow the image tokens and text tokens of the same pair to attend to each other
                    attention_mask[b, i*(LI+LT+self.num_cls_tokens):(i+1)*(LI+LT+self.num_cls_tokens), i*(LI+LT+self.num_cls_tokens):(i+1)*(LI+LT+self.num_cls_tokens)] = True
        elif masking_strategy == "all":
            "all tokens, including cls, can attend to all other tokens, except padding"
            attention_mask = torch.ones(B, N*(LI+LT+self.num_cls_tokens), N*(LI+LT+self.num_cls_tokens), dtype=torch.bool)
            for b in range(B):
                for m in range(N):
                    # find from which item the padding starts
                    if not image_masks[b][m]:
                        attention_mask[b, :, m*(LI+LT+self.num_cls_tokens):] = False
                        break
        else:
            raise NotImplementedError("Masking strategy not implemented")
        return attention_mask

    def forward(self, image_embeds, image_masks, text_embeds, text_masks, timestep=None):
        """
        Args:
            image_embeds: torch.Tensor of shape (batch_size, sequence_length, in_channels)
            image_masks: torch.Tensor of shape (batch_size, sequence_length)
            text_embeds: torch.Tensor of shape (batch_size, sequence_length, in_channels)
            text_masks: torch.Tensor of shape (batch_size, sequence_length)
        """
        B, N, LI, C = image_embeds.shape
        _, _, LT, _ = text_embeds.shape
        # prepare masks
        attention_masks = []
        for l in range(self.num_layers):
            if self.masking_strategy == "modality":
                attention_masks.append(self.prepare_attention_mask(image_masks, text_masks, LI, LT, masking_strategy="modality").to(image_embeds.device)) 
            elif self.masking_strategy == "pair":
                attention_masks.append(self.prepare_attention_mask(image_masks, text_masks, LI, LT, masking_strategy="pair").to(image_embeds.device))
            elif self.masking_strategy == "compression":
                attention_masks.append(self.prepare_attention_mask(image_masks, text_masks, LI, LT, masking_strategy="compression").to(image_embeds.device))
            elif self.masking_strategy == "all":
                attention_masks.append(self.prepare_attention_mask(image_masks, text_masks, LI, LT, masking_strategy="all").to(image_embeds.device))
            else:
                raise NotImplementedError("Masking strategy not implemented")
            
        # concat image and text
        if self.masking_strategy == "compression" or self.masking_strategy == "all":
            # with cls tokens
            batch_cls_tokens = self.cls_tokens.repeat(B, N, 1, 1)
            x = torch.cat([batch_cls_tokens, image_embeds, text_embeds], dim=2)
        else:
            x = torch.cat([image_embeds, text_embeds], dim=2)
        _, _, L, C = x.shape
        if self.masking_strategy == "pair":
            # collapse dim 0 and 1 (pairs as batch items)
            x = x.reshape(B*N, L, C)
        else:
            # collapse dim 1 and 2
            x = x.reshape(B, N*L, C)
        
        # normalize the channels
        x = x.permute(0, 2, 1) # B, C, N*L
        x = self.norm(x)
        x = x.permute(0, 2, 1) # B, N*L, C
        # projection if necessary
        if self.with_in_projection:
            x = self.in_proj(x)
        for attn_mask, block in zip(attention_masks, self.transformer_blocks):
            x = block(hidden_states=x, attention_mask=attn_mask, encoder_attention_mask=attn_mask, timestep=timestep)
        # this returns a B, N*L, C tensor
        if self.with_out_projection:
            x = self.proj_out(x)
        # restore to original dimensions
        x = x.reshape(B, N, L, C)
        # x = x + residual # NOTE: do we want residuals?
        if self.masking_strategy == "compression" or self.masking_strategy == "all":
            x = x[:, :, :self.num_cls_tokens, :]
        # do pooling keeping in mind the masking
        if self.fusion_strategy == "mean":
            pair_embeds = []
            for b in range(B):
                # select only items that are not masked
                selector = torch.ones((N), dtype=torch.bool).to(x.device)
                for i in range(N):
                    if not image_masks[b][i]:
                        selector[i] = False
                item_embeds = x[b, selector, :, :]
                # do the mean pooling
                item_embeds = item_embeds.mean(dim=0, keepdim=False)
                pair_embeds.append(item_embeds)
            pair_embeds = torch.stack(pair_embeds)
            # pair_embeds: B, L, C
        elif self.fusion_strategy == "deferred":
            pair_embeds = x.reshape(B, -1, C) # B, N*L, C
            # the padding items are masked in the unet cross_attn outside of this module
            
        else:
            raise NotImplementedError("Pooling method not implemented")
        return pair_embeds
    