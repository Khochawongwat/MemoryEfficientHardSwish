import torch.nn.functional as F
from torch import nn
import torch

class HardSwish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return HardSwishImplementation.apply(x)
    
class HardSwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.mul(F.relu6(x.add(3.)).div(6.))

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_x[(x < -3.) | (x > 3.)] = 0
        grad_x[(x >= -3.) & (x <= 3.)] *= x[(x >= -3.) & (x <= 3.)].div(3.).add(1./2.)
        return grad_x