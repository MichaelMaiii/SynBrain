import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def sampler_bwd(model, image, prediction):
    t0 = torch.zeros((image.size(0), 1, 1), device=image.device)
    z0 = image
    score  = model(z0, t0.flatten())
    if prediction == 'v':
        z1_pred = z0 + score
    elif prediction == 'x':
        z1_pred = score
    
    return z1_pred