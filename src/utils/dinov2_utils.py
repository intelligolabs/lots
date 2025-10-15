from transformers import AutoImageProcessor, AutoModel
import torch

def get_dinov2_model(model_type="vits14"):
    """Get DINOv2 model that returns full hidden states"""
    model_map = {
        'vits14': 'facebook/dinov2-small',
        'vitb14': 'facebook/dinov2-base',
        'vitl14': 'facebook/dinov2-large',
        'vitg14': 'facebook/dinov2-giant'
    }
    
    model = AutoModel.from_pretrained(model_map[model_type])
    return model  

def get_feature_dim(model_type):
    """Get feature dimension based on model type"""
    dims = {
        'vits14': 384,
        'vitb14': 768,
        'vitl14': 1024,
        'vitg14': 1536
    }
    return dims[model_type]

def extract_features(image_features, pooling_type='cls'):
    """Extract features using different pooling strategies"""
    # image_features should be last_hidden_states with shape [batch_size, num_patches+1, hidden_dim]
    batch_size = image_features.shape[0]
    
    if pooling_type == 'cls':
        return image_features[:, 0]  # get CLS token
    elif pooling_type == 'avg':
        return torch.mean(image_features[:, 1:], dim=1)  # average over patches
    elif pooling_type == 'max':
        return torch.max(image_features[:, 1:], dim=1)[0]  # max over patches
    elif pooling_type == 'cls_max':
        cls_token = image_features[:, 0]
        max_pool = torch.max(image_features[:, 1:], dim=1)[0]
        return torch.cat([cls_token, max_pool], dim=-1)
    elif pooling_type == 'cls_avg':
        cls_token = image_features[:, 0]
        avg_pool = torch.mean(image_features[:, 1:], dim=1)
        return torch.cat([cls_token, avg_pool], dim=-1)
    else:
        raise ValueError(f"Unknown pooling type: {pooling_type}")

def get_pooling_dim(base_dim, pooling_type):
    """Returns the final feature dimension according to the pooling type"""
    if pooling_type in ['cls', 'avg', 'max']:
        return base_dim
    elif pooling_type in ['cls_max', 'cls_avg']:
        return base_dim * 2
    else:
        raise ValueError(f"Unknown pooling type: {pooling_type}")