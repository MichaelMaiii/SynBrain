from sklearn.decomposition import PCA
# from umap._umap import UMAP
import os
import sys
sys.path.append("/home/bingxing2/ailab/maiweijian/SynBrain/src")
sys.path.append("/home/bingxing2/ailab/maiweijian/SynBrain/src/vae")
sys.path.append("/home/bingxing2/ailab/maiweijian/SynBrain/src/s2n")
sys.path.append("/home/bingxing2/ailab/maiweijian/SynBrain/src/sdxl/")
sys.path.append("/home/bingxing2/ailab/maiweijian/SynBrain/src/sdxl/generative_models")
import argparse
import copy
from copy import deepcopy
import logging

from pathlib import Path
from collections import OrderedDict
import json

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator, DistributedType
from accelerate import DistributedDataParallelKwargs

from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from sit import SiT
from loss import SILoss

from diffusers.models import AutoencoderKL
# import wandb_utils
import wandb
import math
from torchvision.utils import make_grid
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize
import torch.distributed as dist
from dataset import *
from mind_utils import *
from utils import *

from scipy.spatial.distance import euclidean
from umap.umap_ import UMAP

from samplers import sampler_bwd

import signal
signal.signal(signal.SIGHUP, signal.SIG_IGN)

logger = get_logger(__name__)

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

from torchdiffeq import odeint_adjoint as odeint
import matplotlib.pyplot as plt


def compute_retrieval(x_fmri, target, device):
    clip_voxels_norm = nn.functional.normalize(x_fmri.flatten(1), dim=-1)
    clip_target_norm = nn.functional.normalize(target.flatten(1), dim=-1)
    
    labels = torch.arange(len(clip_target_norm)).to(device)
    fwd_percent_correct = topk(batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm),
                                    labels, k=1)
    bwd_percent_correct = topk(batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm),
                                    labels, k=1)
    print(f"Forward top1: {fwd_percent_correct}   Backward top1: {bwd_percent_correct}")
    
def compute_retrieval_fwd(x_fmri, target, device):
    clip_voxels_norm = nn.functional.normalize(x_fmri.flatten(1), dim=-1)
    clip_target_norm = nn.functional.normalize(target.flatten(1), dim=-1)
    
    labels = torch.arange(len(clip_target_norm)).to(device)
    fwd_percent_correct = topk(batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm),
                                    labels, k=1)
    print(f"Forward top1: {fwd_percent_correct}")
    
def compute_retrieval_bwd(x_clip, target, device):
    clip_voxels_norm = nn.functional.normalize(x_clip.flatten(1), dim=-1)
    clip_target_norm = nn.functional.normalize(target.flatten(1), dim=-1)
    
    labels = torch.arange(len(clip_target_norm)).to(device)
    bwd_percent_correct = topk(batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm),
                                    labels, k=1)
    print(f"Backward top1: {bwd_percent_correct}")


def plot_umap(clip_target, aligned_clip_voxels):
    print('umap plotting...')
    combined = np.concatenate((clip_target.flatten(1).detach().cpu().numpy(),
                                aligned_clip_voxels.flatten(1).detach().cpu().numpy()), axis=0)
    reducer = UMAP(random_state=42)
    embedding = reducer.fit_transform(combined)

    batch = int(len(embedding) // 2)
    umap_distance = [euclidean(point1, point2) for point1, point2 in zip(embedding[:batch], embedding[batch:])]
    avg_umap_distance = np.mean(umap_distance)
    print(f"Average UMAP Euclidean Distance: {avg_umap_distance}")

    colors = np.array([[0, 0, 1, .5] for i in range(len(clip_target))])
    colors = np.concatenate((colors, np.array([[0, 1, 0, .5] for i in range(len(aligned_clip_voxels))])))

    fig = plt.figure(figsize=(5, 5))
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=colors)
    plt.title(f"Avg.Euclidean Distance = {avg_umap_distance:.4f}")
    
    return fig

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def load_brain_vae(args):
    from brainvae import BrainVAE
    
    config = load_config("/home/bingxing2/ailab/maiweijian/SynBrain/configs/brainvae.yaml")
    model_config = config["model"]["params"]
    ddconfig = model_config["ddconfig"]
    
    model = BrainVAE(ddconfig=ddconfig,
                        hidden_dim=args.hidden_dim,
                        embed_dim=args.embed_dim,
                        clip_weight=1000,
                        kl_weight=0.001
                        )
    
    model_path = f'/home/bingxing2/ailab/group/ai4neuro/BrainVL/BrainSyn/train_logs/{args.brain_path}/last.pth'
    checkpoint = torch.load(model_path, map_location='cpu')
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


#################################################################################
#                                  Generation Loop                              #
#################################################################################

def main(args):    
    
    device = "cuda"
    set_seed(args.seed)
    
    subj = f"sub{args.valid_sub}"
    # save_path = os.path.join(args.save_path, "evals", subj, args.model_name)
    save_path = os.path.join(args.save_path, "evals", subj, args.save_name)
    os.makedirs(save_path, exist_ok=True)
    
    fmri2img_save_path = os.path.join(save_path, "recon")
    os.makedirs(fmri2img_save_path, exist_ok=True)
    
    #! Load SiT
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    ema = SiT(        
            num_patches=256,
            hidden_size=args.embed_dim,
            depth=args.model_depth,
            num_heads=args.model_head,
            **block_kwargs)
    ckpt_name = 'last.pt'
    ckpt = torch.load(f'{os.path.join(args.ckpt_path, args.model_name)}/{ckpt_name}', map_location='cpu')
    # model.load_state_dict(ckpt['model'])
    ema.load_state_dict(ckpt['ema'])
    global_step = ckpt['steps']
    print(f'Load Checkpoint from {global_step} steps......')
        
    ema.to(device).eval()
    requires_grad(ema, False)
    print("params of SiT:")
    count_params(ema)
    
    #! Load MindEye2 for reconstructed fMRI decoding
    args.mindeye_ckpt = f"final_subj0{args.valid_sub}_pretrained_40sess_24bs"
    voxel_dict = {1:15724, 2:14278, 5:13039, 7:12682}
    args.num_voxels = voxel_dict[args.valid_sub]
    mindeyev2 = load_mindeye2(args)
    requires_grad(mindeyev2, False)
    print("params of mindeyev2:")
    count_params(mindeyev2)
    
    # #! Load SDXL UnClip decoder using CPU, parameter: 4.5B
    diffusion_engine, vector_suffix = load_pretrained_sdxl_unclip()
    requires_grad(diffusion_engine, False)
    print("params of sdxl:")
    count_params(diffusion_engine)
    
    #! Load BrainVAE
    brain_enc = load_brain_vae(args).to(device).eval()
    requires_grad(brain_enc, False)
    print("params of brain encoder:")
    count_params(brain_enc)
    
    #! Load test dataset
    test_dataloader = multisub_clip_test_dataset(args)

    all_recon_fmri = None
    all_recon_fmri2imgs = None
    all_clipvoxels = None
    all_clipvoxels_gt = None
    all_clipvoxels_fwd = None
    all_clipvoxels_fm = None
    all_clipvoxels_recon = None
    all_mse_disc = []
    count_imgs = 0
    count_fmri2imgs = 0
    for x_fmri, z_clip, sub in test_dataloader:
        with torch.no_grad():
            x_fmri = x_fmri.float().unsqueeze(1).to(device)
            x_length = x_fmri.shape[-1]
            z_clip = z_clip.float().to(device)

            z_fmri = brain_enc.encode(x_fmri)
            if args.encoder == 'vae':
                z_fmri = z_fmri.mode()
                
            # #! retrieval -> 300 batch sizes
            compute_retrieval(z_fmri.clone(), z_clip.clone(), device)
            if all_clipvoxels is None:
                all_clipvoxels = z_fmri.cpu()
            else:
                all_clipvoxels = torch.vstack((all_clipvoxels, z_fmri.cpu()))

            # if all_clipvoxels_gt is None:
            #     all_clipvoxels_gt = z_clip.cpu()
            # else:
            #     all_clipvoxels_gt = torch.vstack((all_clipvoxels_gt, z_clip.cpu()))

    #         #! cycle sampling
            print('Using onestep cycle sampling..................')
            start_time = time.time()
            sample_fmri = sampler_bwd(
                    ema,
                    z_clip,
                    args.prediction
                )
            end_time = time.time()
            print("Running Time", end_time - start_time, "s")
            
            # compute_retrieval_bwd
            compute_retrieval_bwd(sample_fmri.clone(), z_fmri.clone(), device)
            if all_clipvoxels_fm is None:
                all_clipvoxels_fm = sample_fmri.clone().cpu()
            else:
                all_clipvoxels_fm = torch.vstack((all_clipvoxels_fm, sample_fmri.clone().cpu()))
            
            #! recon fmri
            # sample_fmri = sample_fmri + torch.randn_like(sample_fmri) * 1
            # recon_fmri = brain_enc.decode(sample_fmri, x_length)
            
            # samples = recon_fmri[:3, 0, :].cpu().numpy()  # shape: [3, 15724]

            # # 可视化
            # plt.figure(figsize=(12, 4))
            # for i in range(3):
            #     plt.plot(samples[i], label=f"Sample {i+1}")
            # plt.title("Reconstructed fMRI Signals (First 3 Samples)")
            # plt.xlabel("Voxel Index")
            # plt.ylabel("Activation")
            # plt.legend()
            # plt.grid(True)
            # plt.tight_layout()

            # # 保存图像为 PNG 文件
            # plt.savefig("recon_fmri_plot.png", dpi=300)  # 也可改为 .pdf 等格式
            # plt.close()
            
            recon_fmri = brain_enc.decode(sample_fmri, x_length)
            if all_recon_fmri is None:
                all_recon_fmri = recon_fmri.cpu()
            else:
                all_recon_fmri = torch.vstack((all_recon_fmri, recon_fmri.cpu()))
            
            #! recon fmri retrieval
            z_recon_fmri = brain_enc.encode(recon_fmri)
            if args.encoder == 'vae':
                z_recon_fmri = z_recon_fmri.mode()
                
            compute_retrieval(z_recon_fmri.clone(), z_clip.clone(), device)
            if all_clipvoxels_recon is None:
                all_clipvoxels_recon = z_recon_fmri.cpu()
            else:
                all_clipvoxels_recon = torch.vstack((all_clipvoxels_recon, z_recon_fmri.cpu()))
                
            #! compute mse betweeen original and generated fmri
            mse_disc = F.mse_loss(x_fmri, recon_fmri)
            all_mse_disc.append(mse_disc.item())
            
            #! mindeye generation
            recon_fmri_norm = mindeye_normalize(recon_fmri, args.valid_sub)
            print(torch.mean(recon_fmri_norm), torch.std(recon_fmri_norm))
            sample_fmri2img = mindeyev2_generate(mindeyev2, recon_fmri_norm, args)
                
            for i in range(len(sample_fmri2img)):
                recon_fmri2img = unclip_recon(sample_fmri2img[i].unsqueeze(0).to("cuda:1"),
                            diffusion_engine,
                            vector_suffix,
                            num_samples=1)
                
                if all_recon_fmri2imgs is None:
                    all_recon_fmri2imgs = recon_fmri2img.cpu()
                else:
                    all_recon_fmri2imgs = torch.vstack((all_recon_fmri2imgs, recon_fmri2img.cpu()))
                
                count_fmri2imgs += 1
                if args.save_imgs:
                    recon_fmri2img_resized = transforms.Resize((256, 256))(transforms.ToPILImage()(recon_fmri2img[0]))
                    recon_fmri2img_resized.save(f"{fmri2img_save_path}/{count_fmri2imgs}.png")
                    print(f"Generating {count_fmri2imgs}/1000 images......")
                    
    avg_mse_disc = np.mean(all_mse_disc)
    print(f"Average MSE distance between original and reconstructed fmri: {avg_mse_disc}")
    
    # # resize
    imsize = 256
    # all_recon_imgs = transforms.Resize((imsize,imsize))(all_recon_imgs).float()
    all_recon_fmri2imgs = transforms.Resize((imsize,imsize))(all_recon_fmri2imgs).float()

    # saving
    print(all_recon_fmri2imgs.shape)
    # print(all_recon_imgs.shape)
    # # You can find the all_images file on huggingface: https://huggingface.co/datasets/pscotti/mindeyev2/tree/main/evals
    # torch.save(all_images,"evals/all_images.pt")
    setting_name = args.setting_name
    torch.save(all_recon_fmri2imgs,f"{save_path}/{setting_name}_all_recon_mindeye2.pt")
    torch.save(all_recon_fmri,f"{save_path}/{setting_name}_all_recon_fmri.pt")
    torch.save(all_clipvoxels,f"{save_path}/{setting_name}_all_clipvoxels.pt")
    torch.save(all_clipvoxels_fm,f"{save_path}/{setting_name}_all_clipvoxels_fm.pt")
    torch.save(all_clipvoxels_recon,f"{save_path}/{setting_name}_all_clipvoxels_recon.pt")
    # torch.save(all_clipvoxels_gt,f"/home/bingxing2/ailab/group/ai4neuro/BrainVL/BrainSyn/evals/all_clipvoxels.pt")
    print(f"saved {args.model_name} outputs!")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Evaluation")

    parser.add_argument("--ckpt-path", type=str, default="/home/bingxing2/ailab/group/ai4neuro/BrainVL/BrainSyn/train_logs")
    parser.add_argument("--save-path", type=str, default="/home/bingxing2/ailab/group/ai4neuro/BrainVL/BrainSyn")
    parser.add_argument("--encoder", type=str, default="vae", choices=["mlp", "conv", "vae"])
    parser.add_argument("--brain-path", type=str, default="vae-nsd-s1-vs1-bs24-350")
    parser.add_argument("--setting-name", type=str, default="single_s1_vs1")
    parser.add_argument("--valid-sub", type=int, default=1) #!记得改
    parser.add_argument("--hidden-dim", type=int, default=4096) #!记得改
    parser.add_argument("--embed-dim", type=int, default=1664) #!记得改

    parser.add_argument("--prediction", type=str, default="x")
    parser.add_argument("--model-name", type=str, default="try-fm-vae-s1-vs1-d8-h13-bs24-es350-x-mode")
    parser.add_argument("--save-name", type=str, default="tryfm-vae-s1-vs1-d8-h13-bs24-es350-x-mode")
    parser.add_argument("--model-depth", type=int, default=8)
    parser.add_argument("--model-head", type=int, default=13)  #!记得改
    parser.add_argument("--save_imgs", action=argparse.BooleanOptionalAction, default=True)  #! change to False
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)  #! change to False
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)

    # dataset
    parser.add_argument("--test-batch-size", type=int, default=100)
    parser.add_argument("--data-path", type=str, default="/home/bingxing2/ailab/group/ai4neuro/BrainVL/data/processed_data")

    # # precision
    # parser.add_argument("--allow-tf32", action="store_true")
    # parser.add_argument("--mixed-precision", type=str, default="no", choices=["no", "fp16", "bf16"])

    # seed
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-log", action=argparse.BooleanOptionalAction, default=False)


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args

if __name__ == "__main__":
    args = parse_args()
    
    main(args)
