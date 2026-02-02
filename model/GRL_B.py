import math
import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Optional

class _GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha: float):
        ctx.alpha = float(alpha)
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

class GRL(nn.Module):
    def __init__(self, alpha: float = 0.0):
        super().__init__()
        self.alpha = float(alpha)
    def forward(self, x, alpha: Optional[float] = None):
        a = self.alpha if alpha is None else float(alpha)
        return _GRL.apply(x, a)
    def update_alpha(self, alpha: float):
        self.alpha = float(alpha)

def dann_lambda(progress: float, gamma: float = 10.0) -> float:
    return 2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0

class DomainHead(nn.Module):
    def __init__(self, emb_dim: int, num_domains: int, hidden: int = 128, alpha: float = 0.0, dropout: float = 0.1):
        super().__init__()
        self.grl = GRL(alpha=alpha)
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.PReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, num_domains),
        )
    def forward(self, emb, alpha: Optional[float] = None):
        z = self.grl(emb, alpha=alpha)
        return self.net(z)

class ExtractorWithGRL(nn.Module):
    """Wraps your ResNet extractor (returns 512-D L2-normalized embedding)."""
    def __init__(self, extractor: nn.Module, *, emb_dim: int = 512,
                 num_domains: int = 3, hidden: int = 128, dropout: float = 0.1, grl_alpha: float = 0.0):
        super().__init__()
        self.extractor = extractor
        self.domain_head = DomainHead(emb_dim, num_domains, hidden=hidden, alpha=grl_alpha, dropout=dropout)
    def forward(self, x, *, alpha: Optional[float] = None):
        emb = self.extractor(x)
        dom_logits = self.domain_head(emb, alpha=alpha)
        return emb, dom_logits