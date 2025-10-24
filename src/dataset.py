from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import argparse
import os
import PIL
from PIL import Image
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset, Sampler
import torchvision.transforms as tvtrans
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('param counts:\n{:,} total\n{:,} trainable'.format(total, trainable))

class fMRI_feature_Dataset(Dataset):
    def __init__(self, all_train_fmri, all_train_feat):
        self.fmri_data = []
        self.feat_data = []   
        self.subject_ids = []  

        for subj in all_train_fmri.keys():
            self.fmri_data.extend(all_train_fmri[subj]) 
            self.feat_data.extend(all_train_feat[subj])  
            self.subject_ids.extend([subj] * len(all_train_fmri[subj]))  
            
        print(f'Dataset samples: fmri-{len(self.fmri_data)} feature-{len(self.feat_data)}')

    def __getitem__(self, idx):
        fmri = self.fmri_data[idx]
        clip_feat = self.feat_data[idx]
        subject_id = self.subject_ids[idx] 
        return fmri, clip_feat, subject_id

    def __len__(self):
        return len(self.fmri_data)
    

# 确保每个 batch 中的数据来自同一个受试者
class SubjectSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=False):
            
        self.dataset = dataset
        self.batch_size = batch_size
        self.subject_ids = np.array(self.dataset.subject_ids)
        self.unique_subjects = np.unique(self.subject_ids)
        self.drop_last = drop_last 

    def __iter__(self):
        batches = []
        for subject in self.unique_subjects:
            indices = np.where(self.subject_ids == subject)[0]
            np.random.shuffle(indices)  
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i+self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)  
        
        np.random.shuffle(batches)  
        return iter(batches)

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size 


def multisub_clip_dataset(args):

    all_train_fmri = {}
    all_val_fmri = {}
    all_test_fmri = {}
    all_train_clip = {}
    all_val_clip = {}
    all_test_dino = {}

    subject_list = json.loads(args.subject)
    for subj in subject_list:
        if subj != args.unseen_sub:
            train_fmri = np.load(os.path.join(args.data_path, 'subj0{}/nsd_train_fmri_all_scale_sub{}.npy'.format(subj, subj))).astype(np.float32)
            print(np.max(train_fmri), np.min(train_fmri))
            train_fmri = train_fmri[:750*args.hour]
            all_train_fmri[subj] = train_fmri
            print(f'Train Voxel Sub{subj}: {train_fmri.shape}')
            del train_fmri
            
            train_clip_path = os.path.join(args.data_path, 'subj0{}/nsd_sdxl_clip_train_sub{}.npy'.format(subj, subj))
            train_clip = np.load(train_clip_path).astype(np.float32)
            if subj in [1, 2, 5, 7]:
                train_clip = np.array([feat for feat in train_clip for _ in range(3)])
            train_clip = train_clip[:750*args.hour]
            all_train_clip[subj] = train_clip
            print(f'Train Clip Sub{subj}: {train_clip.shape}')
            del train_clip
            
            #! only test on Subj 1 2 5 7 and average three trials
            if subj == args.valid_sub:
                if subj in [1, 2, 5, 7]:
                    val_fmri = np.load(os.path.join(args.data_path, 'subj0{}/nsd_test_fmri_all_scale_sub{}.npy'.format(subj, subj))).astype(np.float32)  #, mmap_mode='r'
                    val_fmri = val_fmri.reshape(-1, 3, val_fmri.shape[1]).mean(axis=1)  #! avg 3 trials
                    
                print(np.max(val_fmri), np.min(val_fmri))
                all_val_fmri[subj] = val_fmri
                print(f'Valid Voxel Sub{subj}: {val_fmri.shape}')
                del val_fmri
            
                val_clip_path = os.path.join(args.data_path, 'subj0{}/nsd_sdxl_clip_test_sub{}.npy'.format(subj, subj))
                val_clip = np.load(val_clip_path).astype(np.float32)
                all_val_clip[subj] = val_clip
                print(f'Valid Clip Sub{subj}: {val_clip.shape}')
                del val_clip
    

    print(f'Training batch size: {args.local_batch_size}')
    train_dataset = fMRI_feature_Dataset(all_train_fmri, all_train_clip)
    train_subject_sampler = SubjectSampler(train_dataset, batch_size=args.local_batch_size, drop_last=True)
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_subject_sampler)
    # train_dataloader = DataLoader(train_dataset, batch_sampler=train_subject_sampler, num_workers=8, pin_memory=True)

    test_batch_size = 300
    print(f'Testing batch size: {test_batch_size}')
    val_dataset = fMRI_feature_Dataset(all_val_fmri, all_val_clip)
    val_dataloader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, drop_last=False)
    # val_subject_sampler = SubjectSampler(val_dataset, batch_size=test_batch_size, drop_last=True)
    # val_dataloader = DataLoader(val_dataset, batch_sampler=val_subject_sampler)
    # val_dataloader = DataLoader(val_dataset, batch_sampler=val_subject_sampler, num_workers=8, pin_memory=True)
    
    del all_train_fmri, all_val_fmri
    del all_train_clip, all_val_clip

    print("\nDone with Data preparations!")

    return train_dataloader, val_dataloader

def multisub_clip_test_dataset(args):

    all_val_fmri = {}
    all_val_clip = {}

    subj = args.valid_sub
    assert subj in [1, 2, 5, 7]
    
    if subj in [1, 2, 5, 7]:
        val_fmri = np.load(os.path.join(args.data_path, 'subj0{}/nsd_test_fmri_all_scale_sub{}.npy'.format(subj, subj))).astype(np.float32)
        val_fmri = val_fmri.reshape(-1, 3, val_fmri.shape[1]).mean(axis=1)  #! avg 3 trials

    print(np.max(val_fmri), np.min(val_fmri))
    print(np.mean(val_fmri), np.std(val_fmri))
    all_val_fmri[subj] = val_fmri
    print(f'Valid Voxel Sub{subj}: {val_fmri.shape}')
    del val_fmri

    val_clip_path = os.path.join(args.data_path, 'subj0{}/nsd_sdxl_clip_test_sub{}.npy'.format(subj, subj))
    val_clip = np.load(val_clip_path)
    all_val_clip[subj] = val_clip.astype(np.float32)
    print(f'Valid Clip Sub{subj}: {val_clip.shape}')
    del val_clip

    test_batch_size = args.test_batch_size
    print(f'Testing batch size: {test_batch_size}')
    val_dataset = fMRI_feature_Dataset(all_val_fmri, all_val_clip)
    val_dataloader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, drop_last=False)
    
    del all_val_fmri
    del all_val_clip

    print("\nDone with Data preparations!")

    return val_dataloader