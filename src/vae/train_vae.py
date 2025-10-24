import copy
import os
import sys
sys.path.append('/home/bingxing2/ailab/maiweijian/SynBrain/src')
sys.path.append('/home/bingxing2/ailab/maiweijian/SynBrain/src/vae')
import torch
import torch.nn as nn
import argparse
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import logging
import yaml
import random
import wandb
from tqdm import tqdm
from io import BytesIO
from datetime import datetime
import torchvision.utils as vutils

from utils import *
from dataset import *
from mind_utils import *
from brainvae import BrainVAE

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def log_recon_images(sample_image_ema, sample_image_test, epoch):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(vutils.make_grid(sample_image_ema, normalize=True, value_range=(-1, 1)).permute(1, 2, 0).cpu().numpy())
    axes[0].set_title('Recon Image')
    axes[0].axis('off')

    axes[1].imshow(vutils.make_grid(sample_image_test, normalize=True, value_range=(-1, 1)).permute(1, 2, 0).cpu().numpy())
    axes[1].set_title('Test Image')
    axes[1].axis('off')

    plt.tight_layout()
    # plt.close(fig)
    
    return wandb.Image(fig, caption=f"Epoch {epoch}")

def main(args):
    seed_everything(args.seed)
    
    config = load_config("configs/brainvae.yaml")
    lr = args.base_lr
    model_config = config["model"]["params"]
    ddconfig = model_config["ddconfig"]
    chconfig = ddconfig["ch_mult"]
    print(f"Using Layers: {chconfig}")
    
    timestamp = datetime.now().strftime("%m%d%H%M")
    outdir = os.path.abspath(f'{args.save_path}/train_logs/{args.model_name}')
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        
    args.local_batch_size = args.batch_size
    train_dataloader, val_dataloader = multisub_clip_dataset(args)
   
    device = 'cuda'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    device_id = [0, 1]
    
    model = BrainVAE(ddconfig=ddconfig,
                        clip_weight=args.clip_weight,
                        kl_weight=args.kl_weight,
                        hidden_dim=1024,
                        linear_dim=2048,
                        embed_dim=1664
                        )
    
    print("params of BrainAutoencoder")
    count_params(model)
    
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=device_id)
    
    if args.wandb_log:
        if args.resume:
            wandb.login(host='https://api.wandb.ai')
            wandb.init(project="SynBrain", name=args.model_name, config=args, resume="allow", id=args.resume_id)
        else:
            wandb.login(host='https://api.wandb.ai')
            wandb.init(project="SynBrain", name=args.model_name, config=args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=0.05)  # Include brain encoder params

    init_step = 0
    if args.resume and os.path.exists(os.path.join(outdir, 'last.pth')):
        checkpoint_file = os.path.join(outdir, 'last.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # scheduler.load_state_dict(checkpoint["scheduler"])
        init_step = checkpoint["epoch"]+1
        print("=> resume checkpoint (iterations {})".format(checkpoint["epoch"]))
        del checkpoint
        
    # print("Trainable parameters: ")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    
    print(f"{args.model_name} starting with epoch {init_step+1} / {args.num_epochs}")
    progress_bar = tqdm(range(init_step, args.num_epochs), ncols=600)
    
    epoch_rec_losses = []
    losses, val_losses, test_losses, lrs = [], [], [], []
    rec_losses, val_rec_losses, test_rec_losses = [], [], []
    # cycle_losses, val_cycle_losses, test_cycle_losses = [], [], []
    clip_losses, val_clip_losses, test_clip_losses = [], [], []
    kl_losses, val_kl_losses, test_kl_losses = [], [], []
    best_val_loss = 1e9
    for epoch in range(init_step, args.num_epochs):
        print(f"Training epoch {epoch}.....................")
        sims_base = 0.
        val_sims_base = 0.
        test_sims_base = 0.
        fwd_percent_correct = 0.
        bwd_percent_correct = 0.
        val_fwd_percent_correct = 0.
        val_bwd_percent_correct = 0.
        test_fwd_percent_correct = 0.
        test_bwd_percent_correct = 0.
        
        torch.cuda.empty_cache()
        
        model.train()
        for train_i, (fmri, z, sub_id) in enumerate(train_dataloader):
            optimizer.zero_grad()
            
            fmri = fmri.unsqueeze(1).float().to(device)
            z = z.float().to(device)
            
            zs, recon, rec_loss, kl_loss, clip_loss, loss = model(fmri, z, sample_posterior=True)
            
            loss = loss.mean()
            check_loss(loss)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            rec_losses.append(rec_loss.mean().item())
            kl_losses.append(kl_loss.mean().item())
            clip_losses.append(clip_loss.mean().item())

            lrs.append(optimizer.param_groups[0]['lr'])
            # scheduler.step()
            
            zs_norm = nn.functional.normalize(zs.flatten(1), dim=-1)
            z_norm = nn.functional.normalize(z.flatten(1), dim=-1)
            
            sims_base += nn.functional.cosine_similarity(z_norm, zs_norm).mean().item()

            # forward and backward top 1 accuracy
            labels = torch.arange(len(z_norm)).to(device)
            fwd_percent_correct += topk(batchwise_cosine_similarity(zs_norm, z_norm),
                                              labels, k=1)
            bwd_percent_correct += topk(batchwise_cosine_similarity(z_norm, zs_norm),
                                              labels, k=1)
            
        del z, zs, z_norm, zs_norm
            
        model.eval()
        for val_i, (val_fmri, val_z, val_sub) in enumerate(val_dataloader):
            with torch.no_grad():
                val_fmri = val_fmri.unsqueeze(1).float().to(device)
                val_z = val_z.float().to(device)
                
                val_zs, val_recon, val_rec_loss, val_kl_loss, val_clip_loss, val_loss = model(val_fmri, val_z, sample_posterior=False)
                
                val_loss = val_loss.mean()
                check_loss(val_loss)
            
                val_losses.append(val_loss.item())
                val_rec_losses.append(val_rec_loss.mean().item())
                val_kl_losses.append(val_kl_loss.mean().item())
                val_clip_losses.append(val_clip_loss.mean().item())
                
                val_zs_norm = nn.functional.normalize(val_zs.flatten(1), dim=-1)  
                val_z_norm = nn.functional.normalize(val_z.flatten(1), dim=-1)
                
                val_sims_base += nn.functional.cosine_similarity(val_z_norm, val_zs_norm).mean().item()

                # forward and backward top 1 accuracy
                labels = torch.arange(len(val_z_norm)).to(device)
                val_fwd_percent_correct += topk(batchwise_cosine_similarity(val_zs_norm, val_z_norm),
                                                labels, k=1)
                val_bwd_percent_correct += topk(batchwise_cosine_similarity(val_z_norm, val_zs_norm),
                                                labels, k=1)
            
        del val_z, val_zs, val_z_norm, val_zs_norm
        
        current_rec_loss = np.mean(rec_losses[-(train_i + 1):])
        if current_rec_loss < 350:
            print(f"Early stopping at epoch {epoch} due to train_rec_loss < 350 ({current_rec_loss:.2f})")
            break
        
        logs = {"train/loss": np.mean(losses[-(train_i + 1):]),
                "val/loss": np.mean(val_losses[-(val_i + 1):]),
                "train/rec_loss": np.mean(rec_losses[-(train_i + 1):]),
                "val/rec_loss": np.mean(val_rec_losses[-(val_i + 1):]),
                "train/kl_loss": np.mean(kl_losses[-(train_i + 1):]),
                "val/kl_loss": np.mean(val_kl_losses[-(val_i + 1):]),
                "train/clip_loss": np.mean(clip_losses[-(train_i + 1):]),
                "val/clip_loss": np.mean(val_clip_losses[-(val_i + 1):]),
                "train/lr": lrs[-1],
                "train/cosine_sim_base": sims_base / (train_i + 1),
                "val/cosine_sim_base": val_sims_base / (val_i + 1),
                "train/fwd_pct_correct": fwd_percent_correct / (train_i + 1),
                "train/bwd_pct_correct": bwd_percent_correct / (train_i + 1),
                "val/val_fwd_pct_correct": val_fwd_percent_correct / (val_i + 1),
                "val/val_bwd_pct_correct": val_bwd_percent_correct / (val_i + 1)
                }
        
        progress_bar.set_postfix(**logs)
        
        if (epoch % args.ckpt_interval == 0) or (epoch + 1 == args.num_epochs):
            # Save backup last checkpoint
            print(f'Saving Backup last checkpoint at {epoch} epoch out of {args.num_epochs} epochs...')
            ckpt_path = outdir + f'/last.pth'
            print(f'saving last at {epoch}', flush=True)
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                # 'scheduler': scheduler.state_dict(),
                'train_losses': losses,
                'val_losses': val_losses,
                'test_losses': test_losses,
                'lrs': lrs,
            }, ckpt_path)

        if args.plot_recon:
            recon_fmri = save_fmri_recon_image(fmri, recon)
            val_recon_fmri = save_fmri_recon_image(val_fmri, val_recon)
            logs['train/recon_fmri'] = wandb.Image(recon_fmri, caption="Original vs Reconstructed fMRI data")
            logs['val/recon_fmri'] = wandb.Image(val_recon_fmri, caption="Original vs Reconstructed fMRI data")
            del fmri, val_fmri
            del recon, val_recon
            del recon_fmri, val_recon_fmri
        
        wandb.log(logs) if args.wandb_log else None

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("FM parameters")
    parser.add_argument("--seed", type=int, default=1024, help="seed used for initialization")
    parser.add_argument("--model_ckpt", type=str, default=None, help="Model ckpt to init from")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--model_name", type=str, default="vae-nsd-s1-vs1-bs24-350")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--resume_id", type=str, default=None)
    parser.add_argument("--subject", type=str, default="[1]")
    parser.add_argument("--valid-sub", type=int, default=1)
    parser.add_argument("--unseen-sub", type=int, default=9)
    parser.add_argument("--finetune", type=bool, default=False)
    parser.add_argument("--hour", type=int, default=36)
    parser.add_argument("--save_path", type=str, default="/home/bingxing2/ailab/group/ai4neuro/BrainVL/BrainSyn")

    parser.add_argument("--clip_weight", type=float, default=1000)  #1e3
    parser.add_argument("--kl_weight", type=float, default=0.001)

    parser.add_argument("--data_path", type=str, default="/home/bingxing2/ailab/group/ai4neuro/BrainVL/data/processed_data")
    parser.add_argument("--ckpt_interval", type=int, default=1)
    parser.add_argument("--base_lr", type=float, default=1e-4)
    parser.add_argument("--linear_dim", type=int, default=2048)
    parser.add_argument("--wandb_log", type=bool, default=False)
    parser.add_argument("--plot_recon", type=bool, default=True)

    args = parser.parse_args()
    main(args)
