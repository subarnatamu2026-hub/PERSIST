import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim = -1) * self.gamma.unsqueeze(-2).unsqueeze(0) * self.scale