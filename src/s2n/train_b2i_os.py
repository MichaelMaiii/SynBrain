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

import signal
signal.signal(signal.SIGHUP, signal.SIG_IGN)

logger = get_logger(__name__)

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

from torchdiffeq import odeint_adjoint as odeint
import matplotlib.pyplot as plt

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

def compute_retrieval(x_fmri, target, device):
    clip_voxels_norm = nn.functional.normalize(x_fmri.flatten(1), dim=-1)
    clip_target_norm = nn.functional.normalize(target.flatten(1), dim=-1)
    
    labels = torch.arange(len(clip_target_norm)).to(device)
    fwd_percent_correct = topk(batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm),
                                    labels, k=1)
    bwd_percent_correct = topk(batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm),
                                    labels, k=1)
    logging.info(f"Forward top1: {fwd_percent_correct}   Backward top1: {bwd_percent_correct}")


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


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x 


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


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
                        clip_weight=1000,
                        kl_weight=0.001
                        )
    
    model_path = f'{args.output_dir}/{args.vae_path}/last.pth'
    checkpoint = torch.load(model_path, map_location='cpu')
    # model.load_state_dict(checkpoint['model_state_dict'])
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


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):    
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs]
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
    
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiT(        
            num_patches=256,
            hidden_size=1664,
            depth=args.model_depth,
            num_heads=args.model_head,
            **block_kwargs)
        
    print("params of SiT:")
    count_params(model)
    
    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    
    args.local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader, test_dataloader = multisub_clip_dataset(args)
    
    #! Load BrainVAE
    brain_enc = load_brain_vae(args).to(device).eval()
    requires_grad(brain_enc, False)
    print("params of brain encoder:")
    count_params(brain_enc)
    
    #! Load MindEye2 for reconstructed fMRI decoding
    assert args.valid_sub in [1,2,5,7]
    args.mindeye_ckpt = f"final_subj0{args.valid_sub}_pretrained_40sess_24bs"
    voxel_dict = {1:15724, 2:14278,5:13039, 7:12682}
    args.num_voxels = voxel_dict[args.valid_sub]
    
    mindeyev2 = load_mindeye2(args)
    requires_grad(mindeyev2, False)
    print("params of mindeyev2:")
    count_params(mindeyev2)
    
    #! Load SDXL UnClip decoder using CPU, parameter: 4.5B
    diffusion_engine, vector_suffix = load_pretrained_sdxl_unclip()
    requires_grad(diffusion_engine, False)
    print("params of sdxl:")
    count_params(diffusion_engine)

    loss_fn = SILoss(
        prediction=args.prediction
    )
    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )    
    
    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    
    if args.finetune:
        assert args.fm_path is not None
        ckpt_name = 'last.pt'
        ckpt = torch.load(
            f'{os.path.join(args.output_dir, args.fm_path)}/{ckpt_name}',
            map_location='cpu',
            )
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        finetune_step = ckpt['steps']
        print(f'Finetune from {finetune_step} steps......')
        
        print("Trainable parameters: ")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)
        
        # #! freeze layers
        if args.ft_mode == 'mlp':
            print("-------------------------------------")
            requires_grad(model, False)
            # 设定要释放的block层数
            num_unfreeze = 8
            for name, param in model.named_parameters():
                for i in range(num_unfreeze):
                    # if name.startswith(f'blocks.{i}.adaLN'):
                    if name.startswith(f'blocks.{i}.mlp'):
                        param.requires_grad = True
                        
        elif args.ft_mode == 'attn':
            print("-------------------------------------")
            requires_grad(model, False)
            # 设定要释放的block层数
            num_unfreeze = 8
            for name, param in model.named_parameters():
                for i in range(num_unfreeze):
                    # if name.startswith(f'blocks.{i}.adaLN'):
                    if name.startswith(f'blocks.{i}.attn'):
                        param.requires_grad = True
                
        print("Trainable parameters: ")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

    # resume:
    global_step = 0
    if args.resume:
        ckpt_name = 'last.pt'
        ckpt = torch.load(
            f'{os.path.join(args.output_dir, args.exp_name)}/{ckpt_name}',
            map_location='cpu',
            )
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']
        print(f'Resume from {global_step} steps......')

    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader
    )

    # Unwrap the model
    if hasattr(model, 'module'):
        model = accelerator.unwrap_model(model)
        print('Unwrap the model for multiple process......')

    if args.wandb_log:
        if accelerator.is_main_process:
            tracker_config = vars(copy.deepcopy(args))
            if args.resume_id is not None:
                accelerator.init_trackers(
                    project_name="BrainSyn-FM", 
                    config=tracker_config,
                    init_kwargs={
                        "wandb": {"name": f"{args.exp_name}",
                                "resume": "allow",
                                "id": args.resume_id  
                                }
                    },
                )
            else:
                accelerator.init_trackers(
                    project_name="BrainSyn-FM", 
                    config=tracker_config,
                    init_kwargs={
                        "wandb": {"name": f"{args.exp_name}"}
                    },
                )
        
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Labels to condition the model with (feel free to change):
    sample_batch_size = 24 // accelerator.num_processes
    print(f"Total batch size: {args.batch_size}")
    print(f"Num processes: {accelerator.num_processes}")
    print(f"Local batch size: {args.local_batch_size}")
    print(f"Sample batch size: {sample_batch_size}")

    # process ground-truth CLIP features
    train_fmri, train_clip, train_sub = next(iter(train_dataloader))
    train_length = train_fmri.shape[-1]
    with torch.no_grad():
        train_x = brain_enc.encode(train_fmri.float().unsqueeze(1).to(device))
        if args.encoder == 'vae':
            train_x = train_x.mode()
    train_clip = train_clip.float().to(device)
    compute_retrieval(train_x.clone(), train_clip.clone(), device)
    
    train_fmri_norm = mindeye_normalize(train_fmri.clone(), args.valid_sub).unsqueeze(1)
    print(torch.mean(train_fmri_norm), torch.std(train_fmri_norm))
    train_fmri = train_fmri.unsqueeze(1)
    print("Train fMRI shape: ", train_fmri.shape)
    
    test_fmri, test_clip, test_sub = next(iter(test_dataloader))
    test_fmri = test_fmri[:args.local_batch_size]
    test_clip = test_clip[:args.local_batch_size]
    test_length = test_fmri.shape[-1]
    with torch.no_grad():
        test_x = brain_enc.encode(test_fmri.float().unsqueeze(1).to(device))
        if args.encoder == 'vae':
            test_x = test_x.mode()
    test_clip = test_clip.float().to(device)
    compute_retrieval(test_x.clone(), test_clip.clone(), device)
    
    test_fmri_norm = mindeye_normalize(test_fmri.clone(), args.valid_sub).unsqueeze(1)
    print(torch.mean(test_fmri_norm), torch.std(test_fmri_norm))
    test_fmri = test_fmri.unsqueeze(1)
    print("Test fMRI shape: ", test_fmri.shape)

    for epoch in range(args.epochs):
        model.train()
        for x_fmri, z_clip, sub in train_dataloader:  # x is vae features, y is labels
            
            x_fmri = x_fmri.float().unsqueeze(1).to(device)
            z_clip = z_clip.float().to(device)

            with torch.no_grad():
                z_fmri = brain_enc.encode(x_fmri)
                if args.encoder == 'vae':
                    if args.mode:
                        z_fmri = z_fmri.mode()
                    else:
                        z_fmri = z_fmri.sample()
                

            with accelerator.accumulate(model):
                loss = loss_fn(model, z_clip, z_fmri)
                loss = loss.mean()
                    
                ## optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, model) # change ema function
            
            ### enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1                
            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        # "model": model.module.state_dict(),
                        "model": model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/last.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            if (global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0)):
                from samplers import sampler_bwd
                with torch.no_grad():
                    print('Using onestep cycle sampling..................')
                    sample_x = sampler_bwd(
                            ema,
                            train_clip,
                            args.prediction
                        )
                    
                    train_umap_x = plot_umap(train_x.clone(), sample_x.clone())
                    accelerator.log({"train/umap_x": wandb.Image(train_umap_x)})
                    
                    # Brain decode
                    train_fmri_recon = brain_enc.decode(sample_x.clone(), train_length)
                    accelerator.log({"train/fmri_image": wandb.Image(save_fmri_recon_image(train_fmri, train_fmri_recon))})
                    logging.info("Generating EMA brain samples done.")
                    
                    # mindeye2 generation
                    train_fmri_recon_norm = mindeye_normalize(train_fmri_recon, args.valid_sub)
                    print(torch.mean(train_fmri_recon_norm), torch.std(train_fmri_recon_norm))
                    train_clip_recon = mindeyev2_generate(mindeyev2, train_fmri_recon_norm, args)
                    
                    train_clip_raw = mindeyev2_generate(mindeyev2, train_fmri_norm, args)
                    train_fmri2img = sdxl_recon_combined(diffusion_engine, vector_suffix, train_clip_recon.clone(), train_clip_raw.clone())
                    accelerator.log({"train/mindeye2_fmri2img": wandb.Image(train_fmri2img)})
                    logging.info("Generating Train fMRI2Image EMA samples done.")
                    
                    #!!!! Test sampling.......
                    test_sample_x = sampler_bwd(
                            ema,
                            test_clip,
                            args.prediction
                        )
                    
                    test_umap_x = plot_umap(test_x.clone(), test_sample_x.clone())
                    accelerator.log({"test/umap_x": wandb.Image(test_umap_x)})
                    
                    # Brain decode
                    test_fmri_recon = brain_enc.decode(test_sample_x.clone(), test_length)
                    accelerator.log({"test/fmri_image": wandb.Image(save_fmri_recon_image(test_fmri, test_fmri_recon))})
                    logging.info("Generating Test EMA brain samples done.")
                    
                    test_fmri_recon_norm = mindeye_normalize(test_fmri_recon, args.valid_sub)
                    print(torch.mean(test_fmri_recon_norm), torch.std(test_fmri_recon_norm))
                    test_clip_recon = mindeyev2_generate(mindeyev2, test_fmri_recon_norm, args)
                    
                    test_clip_raw = mindeyev2_generate(mindeyev2, test_fmri_norm, args)
                    test_fmri2img = sdxl_recon_combined(diffusion_engine, vector_suffix, test_clip_recon.clone(), test_clip_raw.clone())
                    accelerator.log({"test/mindeye2_fmri2img": wandb.Image(test_fmri2img)})
                    logging.info("Generating Test fMRI2Image EMA samples done.")

            logs = {
                "loss": accelerator.gather(loss).mean().detach().item(), 
                "grad_norm": accelerator.gather(grad_norm).mean().detach().item()
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    model.eval()  # important! This disables randomized embedding dropout
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--sampling-steps", type=int, default=2500)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--ode-sample",  action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume",  action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume-id", type=str, default=None)
    
    # model
    parser.add_argument("--model-depth", type=int, default=8)
    parser.add_argument("--model-head", type=int, default=13)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)  #! change to False
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)

    # dataset
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--subj", type=int, default=1, choices=[1,2,5,7])
    parser.add_argument("--data-path", type=str, default="/home/bingxing2/ailab/group/ai4neuro/BrainVL/data/processed_data")

    # precision
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="no", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400001)
    parser.add_argument("--checkpointing-steps", type=int, default=1000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--subject", type=str, default="[1]")
    parser.add_argument("--valid-sub", type=int, default=1)
    parser.add_argument("--unseen-sub", type=int, default=2)
    parser.add_argument("--hour", type=int, default=36)
    parser.add_argument("--finetune", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ft_mode", type=str, default="all", choices=["all", "attn", "mlp"])

    # loss
    # parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v", "x"]) # currently we only support v-prediction
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument("--alpha-c", type=float, default=0)
    parser.add_argument("--beta", type=float, default=0)

    parser.add_argument("--wandb-log", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cycle", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vae-path", type=str, default=None)
    parser.add_argument("--fm-path", type=str, default=None)
    parser.add_argument("--encoder", type=str, default="vae", choices=["mlp", "conv", "vae"])
    parser.add_argument("--mode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--softclip", action=argparse.BooleanOptionalAction, default=False)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args

if __name__ == "__main__":
    args = parse_args()
    
    main(args)
