import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


def generalized_dice_loss(prediction, target, epsilon=1e-6, per_instance_loss=False):
    # prediction: logits of shape (B, X, Y, Z, C)
    # target: logits of shape (B, X, Y, Z)
    prediction = F.softmax(prediction, dim=-1)
    prediction = rearrange(prediction, "B X Y Z C -> B (X Y Z) C") #flatten all dimensions except channel/class
    target = F.one_hot(target, num_classes=prediction.shape[-1]).float()
    target = rearrange(target, "B X Y Z C -> B (X Y Z) C")
    w_l = target.sum(-1)
    w_l = 1 / (w_l * w_l).clamp(min=epsilon)
    w_l.requires_grad = False # w_l = w_l.detach()?

    intersect = (prediction * target).sum(-1)
    intersect = intersect * w_l

    denominator = (prediction + target).sum(-1)
    denominator = (denominator * w_l).clamp(min=epsilon)

    if per_instance_loss:
        return 1 - (2 * (intersect.sum(-1) / denominator.sum(-1)))

    return 1 - (2 * (intersect.sum() / denominator.sum()))

def proximity_mask(grid_size, temperature=2.0):
    """Create a proximity mask for reweighting loss based on distance from center."""
    center = (np.array(grid_size) - 1) / 2.0
    coords = np.stack(np.meshgrid(
        np.arange(grid_size[0]), np.arange(grid_size[1]), np.arange(grid_size[2]), indexing='ij'), axis=-1)
    dists = np.linalg.norm(coords - center, axis=-1)
    dists = dists / dists.max()
    weights = np.exp(-temperature * dists)
    weights = weights / weights.mean()
    return torch.from_numpy(weights).float()