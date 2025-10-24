import torch
import numpy as np
import torch.nn.functional as F
import lpips
import torch.nn as nn
import math


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))

class SILoss:
    def __init__(
            self,
            prediction='v',
            ):
        self.prediction = prediction

    def __call__(self, model, images, fmri):
        
        assert images.shape == fmri.shape
        
        t0 = torch.zeros((images.size(0), 1, 1), device=images.device)
        z0 = images
        z1 = fmri
        
        if self.prediction == 'v':
            target = z1 - z0
            score  = model(z0, t0.flatten())
            denoise_loss = mean_flat((score - target) ** 2).mean()
            pred_z1 = z0 + score
            
        elif self.prediction == 'x':
            pred_z1 = model(z0, t0.flatten())
            denoise_loss = mean_flat((pred_z1 - z1) ** 2).mean()

        return denoise_loss