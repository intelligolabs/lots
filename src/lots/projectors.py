import torch

class TokenProjector(torch.nn.Module):
    """
    Projection Model
    Takes in input embeddings of shape (BS, L, embeddings_dim) and projects them to (BS, L, cross_attention_dim)
    """

    def __init__(self, embeddings_dim=1024, cross_attention_dim=1024):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.proj = torch.nn.Linear(embeddings_dim, cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, token_embeds):
        """
        token_embeds: torch.Tensor of shape (BS, L, embeddings_dim)

        returns: torch.Tensor of shape (BS, L, attention_dim)
        """
        # image embeds in shape (BS, L, C)
        embeds = token_embeds
        projected_tokens = self.proj(embeds)
        projected_tokens = self.norm(projected_tokens)
        return projected_tokens
    
class SequenceProjModel(torch.nn.Module):
    """
    Projection Model
    Extends a single token to a sequence of tokens
    """

    def __init__(self, cross_attention_dim=1024, embeddings_dim=1024, extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.extra_context_tokens = extra_context_tokens
        self.proj = torch.nn.Linear(embeddings_dim, self.extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, token_embeds):
        embeds = token_embeds
        B, L, C = embeds.shape
        extra_context_tokens = self.proj(embeds).reshape(
            B, L, self.extra_context_tokens, self.cross_attention_dim
        )
        extra_context_tokens = self.norm(extra_context_tokens)
        return extra_context_tokens