import torch
from torch.nn import Module, NLLLoss, Softmax
import torch.nn.functional as F
from torch.autograd import Function

EPS = 1e6

def sparse_softmax(X, dim: int = -1, train=True):
    mask = X < torch.mean(X, dim=dim, keepdim=True)
    mask_offset = mask * (EPS if train else float("Inf"))
    probs = F.softmax(X - mask_offset, dim=dim)
    return probs


class LogSparseSoftmax(Module):
    def __init__(self, dim: int = -1):
        super(LogSparseSoftmax, self).__init__()
        self.dim = dim
    
    def forward(self, X):
        mask = X < torch.mean(X, dim=self.dim, keepdim=True)
        log_probs = F.log_softmax(X - EPS * mask, dim=self.dim)
        return log_probs


class SparseSoftmaxLoss(Module):
    def __init__(self, reduction: str = 'none', dim: int = -1):
        super(SparseSoftmaxLoss, self).__init__()
        self.log_sparse_softmax = LogSparseSoftmax(dim)
        self.reduction = reduction
        self.dim = dim
    
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        return F.nll_loss(self.log_sparse_softmax(input), target, reduction=self.reduction)
