import yaml
import os
import torch
import random
import numpy as np
from io import BytesIO
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from mindeye2 import BrainNetwork, PriorNetwork, BrainDiffusionPrior
import torch.nn as nn
import torch.nn.functional as F

    
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

from torchdiffeq import odeint_adjoint as odeint
import matplotlib.pyplot as plt 

import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import pandas as pd
import seaborn as sns

def evaluate_fmri_reconstruction(all_x_fmri, all_recon_fmri):
    
    # 确保输入是numpy数组（如果是PyTorch张量则转换）
    if isinstance(all_x_fmri, torch.Tensor):
        all_x_fmri = all_x_fmri.detach().cpu().numpy()
    if isinstance(all_recon_fmri, torch.Tensor):
        all_recon_fmri = all_recon_fmri.detach().cpu().numpy()
    
    # 调整形状：移除 singleton 维度，变为 [batch_size, num_voxels]
    all_x_fmri = np.squeeze(all_x_fmri)  # 形状变为 [b, 15724]
    all_recon_fmri = np.squeeze(all_recon_fmri)  # 形状变为 [b, 15724]
    
    # 计算总体MSE
    mse = np.mean((all_x_fmri - all_recon_fmri) **2)
    print(f"总体MSE: {mse:.6f}")
    
    # 计算总体解释方差
    total_var = np.var(all_x_fmri)
    residual_var = np.var(all_x_fmri - all_recon_fmri)
    explained_variance_total = 1 - (residual_var / total_var) if total_var != 0 else 0.0
    print(f"总体解释方差: {explained_variance_total:.4f}")
    
    # 计算每个体素的指标
    num_voxels = all_x_fmri.shape[1]
    spearman_corr = np.zeros(num_voxels)
    explained_variance = np.zeros(num_voxels)
    
    for voxel in range(num_voxels):
        # 提取该体素在所有样本中的原始值和重建值
        x_voxel = all_x_fmri[:, voxel]
        recon_voxel = all_recon_fmri[:, voxel]
        
        # 计算Spearman相关系数
        corr, _ = spearmanr(x_voxel, recon_voxel)
        spearman_corr[voxel] = corr
        
        # 计算解释方差 (避免除以零)
        var_x = np.var(x_voxel)
        if var_x == 0:
            explained_variance[voxel] = 0.0  # 无变异的体素无法被解释
        else:
            var_residual = np.var(x_voxel - recon_voxel)
            explained_variance[voxel] = 1 - (var_residual / var_x)
    
    # 计算体素级Spearman相关系数的统计量
    median_corr = np.median(spearman_corr)
    p90_corr = np.percentile(spearman_corr, 90)
    p95_corr = np.percentile(spearman_corr, 95)
    p99_corr = np.percentile(spearman_corr, 99)
    
    # print(f"\n体素级Spearman相关系数统计:")
    # print(f"中位数 (Median): {median_corr:.4f}")
    # print(f"90% 分位数: {p90_corr:.4f}")
    # print(f"95% 分位数: {p95_corr:.4f}")
    # print(f"99% 分位数: {p99_corr:.4f}")
    
    # # 计算体素级解释方差的统计量
    median_ev = np.median(explained_variance)
    p90_ev = np.percentile(explained_variance, 90)
    p95_ev = np.percentile(explained_variance, 95)
    p99_ev = np.percentile(explained_variance, 99)
    
    # print(f"\n体素级解释方差统计:")
    # print(f"中位数 (Median): {median_ev:.4f}")
    # print(f"90% 分位数: {p90_ev:.4f}")
    # print(f"95% 分位数: {p95_ev:.4f}")
    # print(f"99% 分位数: {p99_ev:.4f}")
    
    # 创建2x1子图展示两个指标的分布
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # 绘制体素级Spearman相关系数的直方图
    sns.histplot(spearman_corr, bins=50, kde=True, color='skyblue', ax=ax1)
    ax1.axvline(median_corr, color='red', linestyle='--', label=f'Median: {median_corr:.4f}')
    ax1.axvline(p90_corr, color='green', linestyle='--', label=f'90% Percentile: {p90_corr:.4f}')
    ax1.axvline(p95_corr, color='purple', linestyle='--', label=f'95% Percentile: {p95_corr:.4f}')
    ax1.axvline(p99_corr, color='orange', linestyle='--', label=f'99% Percentile: {p99_corr:.4f}')
    ax1.set_title('Per-voxel Spearman Distribution', fontsize=14)
    ax1.set_xlabel('Spearman Correlation Coefficient', fontsize=12)
    ax1.set_ylabel('Num Voxels', fontsize=12)
    ax1.set_xlim(-1, 1)  # Spearman相关系数范围是[-1, 1]
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 绘制体素级解释方差的直方图
    sns.histplot(explained_variance, bins=50, kde=True, color='salmon', ax=ax2)
    ax2.axvline(median_ev, color='red', linestyle='--', label=f'Median: {median_ev:.4f}')
    ax2.axvline(p90_ev, color='green', linestyle='--', label=f'90% Percentile: {p90_ev:.4f}')
    ax2.axvline(p95_ev, color='purple', linestyle='--', label=f'95% Percentile: {p95_ev:.4f}')
    ax2.axvline(p99_ev, color='orange', linestyle='--', label=f'99% Percentile: {p99_ev:.4f}')
    ax2.set_title('Per-voxel Explained Variance Distribution', fontsize=14)
    ax2.set_xlabel('Explained Variance', fontsize=12)
    ax2.set_ylabel('Num Voxels', fontsize=12)
    ax2.set_xlim(0, 1)  # 解释方差范围是[0, 1]
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()

    # 将图像直接转换为 PIL Image 对象
    buf = BytesIO()  # 创建一个内存缓冲区
    plt.savefig(buf, format='png')  # 将图像保存到缓冲区而非文件
    plt.close()
    buf.seek(0)  # 重置缓冲区游标位置
    pil_image = Image.open(buf)  # 从缓冲区创建 PIL Image 对象
    
    return pil_image

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def load_brain_autoencoder_mlp(args):
    from script.vae.brainvae import BrainEncoder, BrainDecoder, BrainAutoEncoder
    
    encoder = BrainEncoder(in_dim=15724, out_dim=256*1664, clip_size=1664, h=512, n_blocks=2)
    decoder = BrainDecoder(in_dim=256*1664, out_dim=15724, h=512, n_blocks=2)
    
    model = BrainAutoEncoder(encoder=encoder, decoder=decoder, clip_weight=1000)
    
    vae_path = f'{args.output_dir}/{args.vae_path}/last.pth'
    checkpoint = torch.load(vae_path, map_location='cpu')
    # voxel2clip.load_state_dict(checkpoint['model_state_dict'])
    model_state_dict = {
        k.replace('module.', ''): v 
        for k, v in checkpoint['model'].items() 
        if 'module' in k
    }
    model.load_state_dict(model_state_dict)
    checkpoint_epoch = checkpoint['epoch']
    print(f'Load Checkpoint from {checkpoint_epoch} epoch.....')
    del checkpoint
    
    return model

def load_brain_autoencoder(args):
    from script.vae.brainvae import BrainAutoencoderKL2D_joint
    
    config = load_config(args.config)
    model_config = config["model"]["params"]
    ddconfig = model_config["ddconfig"]
    
    model = BrainAutoencoderKL2D_joint(ddconfig=ddconfig,
                        clip_weight=1000,
                        hidden_dim=4096,
                        cycle=False,
                        )
    
    vae_path = f'{args.output_dir}/{args.vae_path}/last.pth'
    checkpoint = torch.load(vae_path, map_location='cpu')
    # voxel2clip.load_state_dict(checkpoint['model_state_dict'])
    model_state_dict = {
        k.replace('module.', ''): v 
        for k, v in checkpoint['model'].items() 
        if 'module' in k
    }
    model.load_state_dict(model_state_dict)
    checkpoint_epoch = checkpoint['epoch']
    print(f'Load Checkpoint from {checkpoint_epoch} epoch.....')
    del checkpoint
    
    return model

def load_brain_vae(args):
    from script.vae.brainvae import BrainVAE_joint
    
    config = load_config(args.config)
    model_config = config["model"]["params"]
    ddconfig = model_config["ddconfig"]
    
    model = BrainVAE_joint(ddconfig=ddconfig,
                        hidden_dim=args.hidden_dim,
                        clip_weight=1000,
                        kl_weight=0.001
                        )
    
    vae_path = f'{args.output_dir}/{args.vae_path}/last.pth'
    checkpoint = torch.load(vae_path, map_location='cpu')
    # voxel2clip.load_state_dict(checkpoint['model_state_dict'])
    model_state_dict = {
        k.replace('module.', ''): v 
        for k, v in checkpoint['model'].items() 
        if 'module' in k
    }
    model.load_state_dict(model_state_dict)
    checkpoint_epoch = checkpoint['epoch']
    print(f'Load BrainVAE Checkpoint from {checkpoint_epoch} epoch.....')
    del checkpoint
    
    return model

def mindeye_normalize(fmri, subj):
    # 加载保存好的标准化参数
    norm_params = np.load(f'/home/bingxing2/ailab/maiweijian/NeuroFlow/FM/mindeye2/norm_mean_scale_sub{subj}.npz')

    norm_mean_train = norm_params['mean']
    norm_scale_train = norm_params['scale']

    # 将mean和scale转为tensor，并放到fmri的device上
    norm_mean_train = torch.tensor(norm_mean_train, dtype=torch.float32, device=fmri.device)
    norm_scale_train = torch.tensor(norm_scale_train, dtype=torch.float32, device=fmri.device)

    # 使用加载的均值和标准差进行标准化
    fmri = (fmri - norm_mean_train) / norm_scale_train
    
    return fmri

def load_brain_vae(args):
    from script.vae.brainvae import BrainVAE_joint
    
    config = load_config("/home/bingxing2/ailab/maiweijian/NeuroFlow/configs/autoencoder_kl_64x64x256.yaml")
    model_config = config["model"]["params"]
    ddconfig = model_config["ddconfig"]
    
    model = BrainVAE_joint(ddconfig=ddconfig,
                        clip_weight=1000,
                        kl_weight=0.001
                        )
    
    vae_path = f'/home/bingxing2/ailab/group/ai4neuro/BrainVL/BrainSyn/train_logs/{args.vae_path}/last.pth'
    checkpoint = torch.load(vae_path, map_location='cpu')
    # voxel2clip.load_state_dict(checkpoint['model_state_dict'])
    model_state_dict = {
        k.replace('module.', ''): v 
        for k, v in checkpoint['model'].items() 
        if 'module' in k
    }
    model.load_state_dict(model_state_dict)
    checkpoint_epoch = checkpoint['epoch']
    print(f'Load BrainVAE Checkpoint from {checkpoint_epoch} epoch.....')
    del checkpoint
    
    return model

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def neuroflow_generate(brain_enc, model, fmri, args, prediction='v'):
    z_fmri = brain_enc.encode(fmri)
    if args.encoder == 'vae':
        z_fmri = z_fmri.mode()
    
    t0 = torch.zeros((z_fmri.size(0), 1, 1), device=z_fmri.device)
    z0 = z_fmri
    fw_score  = model(z0, t0.flatten())
    if prediction == 'v':
        z1_pred = z0 + fw_score
    
    return z1_pred

# 读取 YAML 配置文件
def load_config(config_file):
    with open(config_file, "r") as file:
        return yaml.safe_load(file)
    
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('param counts:\n{:,} total\n{:,} trainable'.format(total, trainable))

class MindEyeModule(nn.Module):
    def __init__(self):
        super(MindEyeModule, self).__init__()
    def forward(self, x):
        return x
    
class RidgeRegression(torch.nn.Module):
    # make sure to add weight_decay when initializing optimizer to enable regularization
    def __init__(self, input_sizes, out_features): 
        super(RidgeRegression, self).__init__()
        self.out_features = out_features
        self.linears = torch.nn.ModuleList([
                torch.nn.Linear(input_size, out_features) for input_size in input_sizes
            ])
    def forward(self, x, subj_idx):
        out = self.linears[subj_idx](x[:,0]).unsqueeze(1)
        return out
    
# def load_mindeye2():
#     # ckpt_path = "/home/bingxing2/ailab/group/ai4neuro/mindeyev2/train_logs/final_subj01_pretrained_40sess_24bs/last.pth"
#     ckpt_path = "/home/bingxing2/ailab/group/ai4neuro/mindeyev2/train_logs/final_multisubject_subj01/last.pth"
#     ckpt = torch.load(ckpt_path)
#     model_state_dict = ckpt['model_state_dict']

#     # Remove unuseful weight or set strict=False
#     fields_to_remove = ['blin1', 'bdropout', 'bnorm', 'bupsampler', 'b_maps_projector', 'diffusion_prior']
#     keys_to_remove = [key for key in model_state_dict.keys() if any(field in key for field in fields_to_remove)]
#     for key in keys_to_remove:
#         del model_state_dict[key]

#     # num_voxels_list = [15724, 14278, 15226, 13153, 13039, 17907, 12682, 14386]
#     num_voxels_list = [14278, 15226, 13153, 13039, 17907, 12682, 14386]  #! Subject 2-8

#     model = MindEyeEncoder()
#     model.ridge = RidgeRegression(num_voxels_list, 4096)
#     model.backbone = Backbone()
#     model.load_state_dict(model_state_dict, strict=True)
    
#     # fake test_data
#     test_voxel = torch.randn(8, 1, 13153)
#     subj_idx = num_voxels_list.index(test_voxel.size(-1))
#     out = model.ridge(test_voxel, subj_idx)
#     print(out.shape)
#     backbone, clip_voxels = model.backbone(out)
#     print(backbone.shape, clip_voxels.shape)
    
#     return model
        
def load_mindeye2(args):
    
    model = MindEyeModule()
    
    clip_seq_dim = 256
    clip_emb_dim = 1664
    hidden_dim = 4096
    # num_voxels = 15724
    num_voxels = args.num_voxels
    num_voxels_list = [15724, 14278, 15226, 13153, 13039, 17907, 14386]
    if args.valid_sub in [1,2,5,7]:
        model.ridge = RidgeRegression([num_voxels], out_features=hidden_dim)
    else:
        model.ridge = RidgeRegression(num_voxels_list, out_features=hidden_dim)

    model.backbone = BrainNetwork(h=hidden_dim, in_dim=hidden_dim, seq_len=1, blurry_recon=False,
                            clip_size=clip_emb_dim, out_dim=clip_emb_dim*clip_seq_dim)
    # count_params(model.ridge)
    # count_params(model.backbone)
    # count_params(model)

    # setup diffusion prior network
    out_dim = clip_emb_dim
    depth = 6
    dim_head = 52
    heads = clip_emb_dim//52 # heads * dim_head = clip_emb_dim
    timesteps = 100

    prior_network = PriorNetwork(
            dim=out_dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            causal=False,
            num_tokens = clip_seq_dim,
            learned_query_mode="pos_emb"
        )

    model.diffusion_prior = BrainDiffusionPrior(
        net=prior_network,
        image_embed_dim=out_dim,
        condition_on_text_encodings=False,
        timesteps=timesteps,
        cond_drop_prob=0.2,
        image_embed_scale=None,
    )
    model.to(device)

    count_params(model.diffusion_prior)
    # count_params(model)

    ckpt_path = f"/home/bingxing2/ailab/group/ai4neuro/mindeyev2/train_logs/{args.mindeye_ckpt}/last.pth"
    print(f"Loading {ckpt_path}......")
    
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    # print(state_dict)
    model.load_state_dict(state_dict, strict=False)
    del checkpoint
    print("ckpt loaded!")
    
    return model

def mindeyev2_generate(model, x, args, return_all=False):
    # x = x[:5].to(device)
    x = x.to(device)
    # assert args.valid_sub != 7
    if args.valid_sub in [1,2,5,7]:
        out = model.ridge(x, 0)
    else:
        if args.valid_sub == 8:
            out = model.ridge(x, args.valid_sub - 2)
        else:
            out = model.ridge(x, args.valid_sub - 1)
    print(out.shape)
    backbone, clip_voxels, _ = model.backbone(out)
    print(backbone.shape, clip_voxels.shape)
    
    prior_out = model.diffusion_prior.p_sample_loop(backbone.shape, 
        text_cond = dict(text_embed = backbone), 
        cond_scale = 1., timesteps = 20)
    
    if return_all:
        return prior_out, clip_voxels
    else:
        return prior_out



def load_pretrained_sdxl_unclip():
    # from sdxl.models import *
    from omegaconf import OmegaConf
    from sdxl.generative_models.sgm.models.diffusion import DiffusionEngine

    # prep unCLIP
    config = OmegaConf.load("/home/bingxing2/ailab/maiweijian/NeuroFlow/sdxl/generative_models/configs/unclip6.yaml")
    config = OmegaConf.to_container(config, resolve=True)
    unclip_params = config["model"]["params"]
    network_config = unclip_params["network_config"]
    denoiser_config = unclip_params["denoiser_config"]
    first_stage_config = unclip_params["first_stage_config"]
    conditioner_config = unclip_params["conditioner_config"]
    sampler_config = unclip_params["sampler_config"]
    scale_factor = unclip_params["scale_factor"]
    disable_first_stage_autocast = unclip_params["disable_first_stage_autocast"]
    offset_noise_level = unclip_params["loss_fn_config"]["params"]["offset_noise_level"]

    first_stage_config['target'] = 'sgm.models.autoencoder.AutoencoderKL'
    sampler_config['params']['num_steps'] = 38

    diffusion_engine = DiffusionEngine(network_config=network_config,
                        denoiser_config=denoiser_config,
                        first_stage_config=first_stage_config,
                        conditioner_config=conditioner_config,
                        sampler_config=sampler_config,
                        scale_factor=scale_factor,
                        disable_first_stage_autocast=disable_first_stage_autocast)
    # set to inference
    diffusion_engine.eval().requires_grad_(False)
    diffusion_engine.to(device)

    ckpt_path = f'/home/bingxing2/ailab/group/ai4neuro/mindeyev2/unclip6_epoch0_step110000.ckpt'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    diffusion_engine.load_state_dict(ckpt['state_dict'])

    batch={"jpg": torch.randn(1,3,1,1).to(device), # jpg doesnt get used, it's just a placeholder
        "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
        "crop_coords_top_left": torch.zeros(1, 2).to(device)}
    out = diffusion_engine.conditioner(batch)
    vector_suffix = out["vector"].to(device)
    print("vector_suffix", vector_suffix.shape)
    
    return diffusion_engine, vector_suffix


def unclip_recon(x, diffusion_engine, vector_suffix,
                 num_samples=1, offset_noise_level=0.04):
    from sdxl.generative_models.sgm.util import append_dims
    assert x.ndim==3
    if x.shape[0]==1:
        x = x[[0]]
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16), diffusion_engine.ema_scope():
        z = torch.randn(num_samples,4,96,96).to(device) # starting noise, can change to VAE outputs of initial image for img2img

        # clip_img_tokenized = clip_img_embedder(image) 
        # tokens = clip_img_tokenized
        token_shape = x.shape
        tokens = x
        c = {"crossattn": tokens.repeat(num_samples,1,1), "vector": vector_suffix.repeat(num_samples,1)}

        tokens = torch.randn_like(x)
        uc = {"crossattn": tokens.repeat(num_samples,1,1), "vector": vector_suffix.repeat(num_samples,1)}

        for k in c:
            c[k], uc[k] = map(lambda y: y[k][:num_samples].to(device), (c, uc))

        noise = torch.randn_like(z)
        sigmas = diffusion_engine.sampler.discretization(diffusion_engine.sampler.num_steps)
        sigma = sigmas[0].to(z.device)

        if offset_noise_level > 0.0:
            noise = noise + offset_noise_level * append_dims(
                torch.randn(z.shape[0], device=z.device), z.ndim
            )
        noised_z = z + noise * append_dims(sigma, z.ndim)
        noised_z = noised_z / torch.sqrt(
            1.0 + sigmas[0] ** 2.0
        )  # Note: hardcoded to DDPM-like scaling. need to generalize later.

        def denoiser(x, sigma, c):
            return diffusion_engine.denoiser(diffusion_engine.model, x, sigma, c)

        samples_z = diffusion_engine.sampler(denoiser, noised_z, cond=c, uc=uc)
        samples_x = diffusion_engine.decode_first_stage(samples_z)
        samples = torch.clamp((samples_x*.8+.2), min=0.0, max=1.0)
        # samples = torch.clamp((samples_x + .5) / 2.0, min=0.0, max=1.0)
        return samples
    
def denorm_z_score(z_norm, sub):
    stats = np.load(f'/home/bingxing2/ailab/maiweijian/NeuroFlow/FM/mindeye2/clip_stats_sub{sub}.npz')
    mu = stats['mu']
    std = stats['std']
    mu = torch.from_numpy(mu).float().to(z_norm.device)
    std = torch.from_numpy(std).float().to(z_norm.device)
    
    z = z_norm * std + mu
    return z


def sdxl_recon_combined_all(diffusion_engine, vector_suffix, sample_clip, train_clip, sample_f2i, args, num_samples=5):
    
    if args.zscore_clip:
        sample_clip = denorm_z_score(sample_clip.detach().to(device), args.valid_sub)
        sample_f2i = denorm_z_score(sample_f2i.detach().to(device), args.valid_sub)
        train_clip = denorm_z_score(train_clip.detach().to(device), args.valid_sub)
    else:
        sample_clip = sample_clip.detach().to(device)
        sample_f2i = sample_f2i.detach().to(device)
        train_clip = train_clip.detach().to(device)
    
    all_recon_f2i = None
    all_recon_imgs = None
    all_raw_imgs = None
    for i in range(num_samples):
        recon_img = unclip_recon(sample_clip[i].unsqueeze(0),
                    diffusion_engine,
                    vector_suffix,
                    num_samples=1)
        
        recon_f2i = unclip_recon(sample_f2i[i].unsqueeze(0),
                    diffusion_engine,
                    vector_suffix,
                    num_samples=1)
        
        raw_img = unclip_recon(train_clip[i].unsqueeze(0),
            diffusion_engine,
            vector_suffix,
            num_samples=1)
        
        if all_recon_imgs is None:
            all_recon_imgs = recon_img.cpu()
            all_recon_f2i = recon_f2i.cpu()
            all_raw_imgs = raw_img.cpu()
        else:
            all_recon_imgs = torch.vstack((all_recon_imgs, recon_img.cpu()))
            all_recon_f2i = torch.vstack((all_recon_f2i, recon_f2i.cpu()))
            all_raw_imgs = torch.vstack((all_raw_imgs, raw_img.cpu()))
            
    # 转换成 PIL 图像
    to_pil = transforms.ToPILImage()

    num_samples = all_raw_imgs.shape[0]  # 样本数
    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 3, 9))  # 3行多列

    for i in range(num_samples):
        # 原图（GT）
        axes[0, i].imshow(to_pil(all_raw_imgs[i]))
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Raw {i+1}', fontsize=10)

        # 重建图（Recon）
        axes[1, i].imshow(to_pil(all_recon_imgs[i]))
        axes[1, i].axis('off')
        axes[1, i].set_title(f'Image {i+1}', fontsize=10)
        
        # 重建图（f2i）
        axes[2, i].imshow(to_pil(all_recon_f2i[i]))
        axes[2, i].axis('off')
        axes[2, i].set_title(f'fMRI2Image {i+1}', fontsize=10)

    # 设置整张图标题
    plt.suptitle("Top: Raw Images | Middle: Recon Image | Bottom: Recon fMRI2Img", fontsize=14)

    plt.tight_layout()
    return fig

def sdxl_recon_combined(diffusion_engine, vector_suffix, sample_clip, train_clip, num_samples=5):
    
    sample_clip = sample_clip.detach().to(device)
    train_clip = train_clip.detach().to(device)
    
    all_recon_imgs = None
    all_raw_imgs = None
    # num_samples = 5
    for i in range(num_samples):
        recon_img = unclip_recon(sample_clip[i].unsqueeze(0),
                    diffusion_engine,
                    vector_suffix,
                    num_samples=1)
        
        raw_img = unclip_recon(train_clip[i].unsqueeze(0),
            diffusion_engine,
            vector_suffix,
            num_samples=1)
        
        if all_recon_imgs is None:
            all_recon_imgs = recon_img.cpu()
            all_raw_imgs = raw_img.cpu()
        else:
            all_recon_imgs = torch.vstack((all_recon_imgs, recon_img.cpu()))
            all_raw_imgs = torch.vstack((all_raw_imgs, raw_img.cpu()))
            
    # 转换成 PIL 图像
    to_pil = transforms.ToPILImage()

    num_samples = all_raw_imgs.shape[0]  # 样本数
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))  # 两行多列

    for i in range(num_samples):
        # 原图（GT）
        axes[0, i].imshow(to_pil(all_raw_imgs[i]))
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Raw {i+1}', fontsize=10)

        # 重建图（Recon）
        axes[1, i].imshow(to_pil(all_recon_imgs[i]))
        axes[1, i].axis('off')
        axes[1, i].set_title(f'Recon {i+1}', fontsize=10)

    # 设置整张图标题
    plt.suptitle("Top: Raw Images | Bottom: Reconstructed Images", fontsize=14)

    plt.tight_layout()
    return fig

def vavae_recon_combined(model, sample_clip, train_clip, num_samples=5):
    
    # b, d, c = sample_clip.shape
    # assert d == 256
    # sample_clip = sample_clip.reshape(b, d//16, d//16, c).permute(0, 3, 1, 2)
    # train_clip = train_clip.reshape(b, d//16, d//16, c).permute(0, 3, 1, 2)
    
    recon = model.decode(sample_clip)
    recon = torch.clamp(127.5 * recon + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
    raw = model.decode(train_clip)
    raw = torch.clamp(127.5 * raw + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
    
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))  # 两行多列

    for i in range(num_samples):
        # 原图（GT）
        axes[0, i].imshow(raw[i])
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Raw {i+1}', fontsize=10)

        # 重建图（Recon）
        axes[1, i].imshow(recon[i])
        axes[1, i].axis('off')
        axes[1, i].set_title(f'Recon {i+1}', fontsize=10)

    # 设置整张图标题
    plt.suptitle("Top: Raw Images | Bottom: Reconstructed Images", fontsize=14)
    plt.tight_layout()
    
    # 将图像直接转换为 PIL Image 对象
    buf = BytesIO()  # 创建一个内存缓冲区
    plt.savefig(buf, format='png')  # 将图像保存到缓冲区而非文件
    plt.close()
    buf.seek(0)  # 重置缓冲区游标位置
    pil_image = Image.open(buf)  # 从缓冲区创建 PIL Image 对象
    
    return pil_image

def get_weight(model):
    size_all_mb = sum(p.numel() for p in model.parameters()) / 1024**2
    return size_all_mb

    
def save_fmri_recon_image(fmri_data, recon_data):
    """
    将原始 fMRI 数据和重建的 fMRI 数据绘制到一张图的左右两侧并记录到 wandb。
    假设 fMRI 数据和重建数据形状为 [batch, 1, length]。
    """
    # 去掉维度为 1 的通道，变为 [batch, length]
    # fmri_data = fmri_data[0].detach().cpu().numpy()
    # recon_data = recon_data[0].detach().cpu().numpy()
    fmri_data = fmri_data[:5].squeeze(1).detach().cpu().numpy()
    recon_data = recon_data[:5].squeeze(1).detach().cpu().numpy()

    # 创建一个包含两个子图的图像 (1x2布局)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # 生成不同的颜色，每个样本用不同的颜色绘制
    colors = plt.cm.viridis(np.linspace(0, 1, fmri_data.shape[0]))
    
    # 绘制左侧：原始 fMRI 数据曲线
    for i in range(fmri_data.shape[0]):
        axes[0].plot(fmri_data[i], color=colors[i], label=f'Sample {i}')
    axes[0].set_title(f"Original fMRI data ({fmri_data.shape[0]} samples)")
    axes[0].set_xlabel("Voxel Index")
    axes[0].set_ylabel("Signal Intensity")

    # 绘制右侧：重建的 fMRI 数据曲线
    for i in range(recon_data.shape[0]):
        axes[1].plot(recon_data[i], color=colors[i], label=f'Sample {i}')
        
    # mse_ = F.mse_loss(recon_data, fmri_data)

    axes[1].set_title(f"Reconstructed fMRI data ({recon_data.shape[0]} samples)")
    axes[1].set_xlabel("Voxel Index")
    axes[1].set_ylabel("Signal Intensity")

    # 将图像直接转换为 PIL Image 对象
    buf = BytesIO()  # 创建一个内存缓冲区
    plt.savefig(buf, format='png')  # 将图像保存到缓冲区而非文件
    plt.close()
    buf.seek(0)  # 重置缓冲区游标位置
    pil_image = Image.open(buf)  # 从缓冲区创建 PIL Image 对象
    
    return pil_image

def save_eeg_recon_image(fmri_data, recon_data):
    """
    将原始 fMRI 数据和重建的 fMRI 数据绘制到一张图的左右两侧并记录到 wandb。
    假设 fMRI 数据和重建数据形状为 [batch, 1, length]。
    """
    # 去掉维度为 1 的通道，变为 [batch, length]
    # fmri_data = fmri_data[0].detach().cpu().numpy()
    # recon_data = recon_data[0].detach().cpu().numpy()
    fmri_data = fmri_data[0].detach().cpu().numpy()
    recon_data = recon_data[0].detach().cpu().numpy()

    # 创建一个包含两个子图的图像 (1x2布局)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # 生成不同的颜色，每个样本用不同的颜色绘制
    colors = plt.cm.viridis(np.linspace(0, 1, fmri_data.shape[0]))
    
    # 绘制左侧：原始 fMRI 数据曲线
    for i in range(fmri_data.shape[0]):
        axes[0].plot(fmri_data[i], color=colors[i], label=f'Sample {i}')
    axes[0].set_title(f"Original fMRI data ({fmri_data.shape[0]} samples)")
    axes[0].set_xlabel("Voxel Index")
    axes[0].set_ylabel("Signal Intensity")

    # 绘制右侧：重建的 fMRI 数据曲线
    for i in range(recon_data.shape[0]):
        axes[1].plot(recon_data[i], color=colors[i], label=f'Sample {i}')
        
    # mse_ = F.mse_loss(recon_data, fmri_data)

    axes[1].set_title(f"Reconstructed fMRI data ({recon_data.shape[0]} samples)")
    axes[1].set_xlabel("Voxel Index")
    axes[1].set_ylabel("Signal Intensity")

    # 将图像直接转换为 PIL Image 对象
    buf = BytesIO()  # 创建一个内存缓冲区
    plt.savefig(buf, format='png')  # 将图像保存到缓冲区而非文件
    plt.close()
    buf.seek(0)  # 重置缓冲区游标位置
    pil_image = Image.open(buf)  # 从缓冲区创建 PIL Image 对象
    
    return pil_image


def check_loss(loss):
    if loss.isnan().any():
        raise ValueError('NaN loss')

def seed_everything(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        ## needs to be False to use conv3D
        print('Note: not using cudnn.deterministic')