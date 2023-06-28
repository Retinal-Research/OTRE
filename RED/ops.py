import numpy as np
import torch
import torch.nn as nn

# from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

# class MS_SSIM_Loss(nn.Module):


class DataFidelityTorch(nn.Module):
    def __init__(self, y, loss_func):
        super().__init__()
        self.y = y
        self.loss_func = loss_func

    def forward(self, x):
        assert isinstance(x, torch.Tensor), "Requires Tensor input & loss_func"
        x.requires_grad_(True) 

        return self.loss_func(x, self.y)

    def grad(self, x, tensor=True):
        loss = self.forward(x)
        gradients = torch.autograd.grad(outputs=loss, inputs=x,
                                        grad_outputs=torch.ones(loss.size()).to(x.device),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        if tensor:
            return gradients
        else:
            return gradients.detach().clone().cpu().numpy()


class Regularizer(nn.Module):
    def __init__(self, enhancer):
        super().__init__()
        self.enhancer = enhancer

    def forward(self, x, tensor=True):
        assert isinstance(self.enhancer, nn.Module) and isinstance(x, torch.Tensor), "Requires Tensor input&enhancer"

        with torch.no_grad():
            x_enhanced = self.enhancer(x)

            if tensor:
                return x - x_enhanced
            else:
                return (x - x_enhanced).detach().clone().cpu().numpy()


# denoiser = Enhancer()
# reg = Regularizer(denoiser)
# reg(torch.randn(1, 3, 256, 256))