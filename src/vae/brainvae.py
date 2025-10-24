import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from module import *
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution_2D


def MLP(hidden_dim, linear_dim, embed_dim):
    mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, linear_dim),
            nn.LayerNorm(linear_dim),
            nn.GELU(),
            nn.Linear(linear_dim, linear_dim),
            nn.LayerNorm(linear_dim),
            nn.GELU(),
            nn.Linear(linear_dim, embed_dim)
            )
    return mlp

class BrainVAE(nn.Module):
    def __init__(self,
                 ddconfig,
                 clip_weight=1,
                 kl_weight=1,
                 hidden_dim=4096,
                 linear_dim=2048,
                 embed_dim=1664,
                 ):
        super().__init__()
        
        self.encoder = NeuroEncoder(**ddconfig)
        self.decoder = NeuroDecoder(**ddconfig)
        
        self.kl_weight = kl_weight
        self.clip_weight = clip_weight
    
        self.pre_projector_mean = MLP(hidden_dim, linear_dim, embed_dim)
        self.pre_projector_logvar = MLP(hidden_dim, linear_dim, embed_dim)
        self.post_projector = MLP(embed_dim, linear_dim, hidden_dim)
            
    def soft_clip_loss(self, preds, targs, temp=0.125):
        
        preds = F.normalize(preds.flatten(1), dim=-1)  # 在最后一维上归一化
        targs = F.normalize(targs.flatten(1), dim=-1)  # 在最后一维上归一化
    
        clip_clip = (targs @ targs.T)/temp
        brain_clip = (preds @ targs.T)/temp
        
        loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        
        loss = (loss1 + loss2)/2
        return loss
        
    def mse_loss(self, reconstructions, inputs):
        loss = nn.functional.mse_loss(reconstructions, inputs, reduction='sum') / inputs.shape[0]
        return loss
    
    def kl_loss(self, posterior, inputs):
        loss = posterior.kl()  #与正态分布计算KL损失
        loss = loss.sum() / inputs.shape[0]
        return loss
    
    def encode(self, x):
        x = self.encoder(x)  #[b, 256, 4096]
        x_mean = self.pre_projector_mean(x)  #[b, 256, 1664]
        x_logvar = self.pre_projector_logvar(x)  #[b, 256, 1664]
        moments = torch.cat((x_mean, x_logvar), dim=1)  # [b, 512, 1664]
        posterior = DiagonalGaussianDistribution_2D(moments)  # 对角高斯采样
        return posterior
    
    def decode(self, x, target_length):
        x = self.post_projector(x)  #[b, 256, 4096]
        x = self.decoder(x, target_length) 
        return x

    def forward(self, x, zs, sample_posterior):
        self.target_length = x.shape[2]
        
        posterior = self.encode(x)  #[b, 256, 1664]
        if sample_posterior:
            z = posterior.sample()  # 采样潜在变量 z
        else:
            z = posterior.mode()  # 使用均值作为潜在变量 z
            
        recon = self.decode(z, self.target_length)
    
        recon_loss = self.mse_loss(recon, x)
        kl_loss = self.kl_loss(posterior, x)

        if self.clip_weight == 0:
            clip_loss = torch.tensor(0.0, device=x.device)
        else:
            clip_loss = self.soft_clip_loss(z, zs)

        loss = recon_loss + kl_loss * self.kl_weight + clip_loss * self.clip_weight
        
        return z, recon, recon_loss, kl_loss, clip_loss, loss