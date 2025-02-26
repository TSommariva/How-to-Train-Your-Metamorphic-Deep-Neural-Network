#/work/tesi_tsommariva/experiments/AaITERATIVE/OnlyLast_OnlyMod4_NoEval_resmlpDictXL_8simultaneusBlocks_50_4AccumulationSteps_warmup_cosine_20e/cifar100_nerf_last.pth
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neumeta.hypermodel import NeRF_MLP_Compose, NeRF_ResMLP_Compose
from neumeta.models import create_model_cifar100 as create_model
from neumeta.utils import (AverageMeter, EMA, create_key_masks, get_cifar100,
                           get_hypernet, get_optimizer,get_optimizer_scaledFT, load_checkpoint,
                           parse_args, print_omegaconf, sample_coordinates, sample_weights, sample_merge_model,
                           sample_subset,  save_checkpoint,
                           set_seed, shuffle_coordiates_all,
                           validate_single, validate_all_dimensions,
                           initialize_wandb,find_max_dim, register_hooks_and_print_shapes, extend_nerf_compose, load_trained_blocks,
                           get_cifar_optimizer)

import wandb
import time
import gc

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def get_num_workers():
    try:
        return int(os.environ.get("SLURM_CPUS_PER_TASK", 2))
    except (ValueError, TypeError):
        return 2

def init_model_dict(args, num_blocks = 1, single_block = False):
    """
    Initializes a dictionary of models for each dimension in the given range, along with ground truth models for the starting dimension.

    Args:
        args: An object containing the arguments for initializing the models.

    Returns:
        dim_dict: A dictionary containing the models for each dimension, along with their corresponding coordinates, keys, indices, size, and ground truth models.
        gt_model_dict: A dictionary containing the ground truth models for the starting dimension.
    """
    dim_dict = {}
    gt_model_dict = {}
    if not args.experiment.iterative:
        num_blocks=args.model.num_param
    for dim in range(args.dimensions.range[1], args.dimensions.range[1] + 1, 4):
        model_cls = create_model(args.model.type, 
                                 hidden_dim=dim, num_param=num_blocks, bottom_up=args.model.bottom_up,single_block=single_block ,
                                 path=args.model.pretrained_path, smooth=args.model.smooth, fuse=args.model.smooth, prior=False)
         
        if device=="cuda" and torch.backends.cudnn.version() >= 7603:
            model_cls.to(device, memory_format=torch.channels_last)  # Module parameters need to be channels last
        else:
            model_cls.to(device)
        
        optimizer = get_cifar_optimizer(args, model_cls)
            
        coords_tensor, keys_list, indices_list, size_list = sample_coordinates(model_cls)
        dim_dict[f"{dim}"] = (model_cls, optimizer, coords_tensor, keys_list, indices_list, size_list, None)
        
        #if device=="cuda" and torch.backends.cudnn.version() >= 7603:
        #    input_tensor = torch.randn(1, 3, 32, 32).to(device, memory_format=torch.channels_last)
        #else:
        #    input_tensor = torch.randn(1, 3, 32, 32).to(device)
        #
        #register_hooks_and_print_shapes(model_cls, input_tensor)

        if dim == args.dimensions.start:
            print(f"Loading model for dim {dim}")
            model_trained = create_model(args.model.type, 
                                 hidden_dim=dim, num_param=num_blocks, bottom_up=args.model.bottom_up, single_block=single_block,
                                 path=args.model.pretrained_path, smooth=args.model.smooth, fuse=args.model.smooth)
            
            if device=="cuda" and torch.backends.cudnn.version() >= 7603:
                model_trained.to(device, memory_format=torch.channels_last)  # Module parameters need to be channels last
            else:
                model_trained.to(device)
            model_trained.eval()
            
            gt_model_dict[f"{dim}"] = model_trained
    return dim_dict, gt_model_dict

args = parse_args()
print_omegaconf(args)
set_seed(args.experiment.seed)
num_workers = get_num_workers()
train_loader, val_loader = get_cifar100(args.training.batch_size, num_workers)
    
model = create_model(args.model.type, 
                     hidden_dim=args.dimensions.start,
                     num_param=args.model.num_param,
                     bottom_up=args.model.bottom_up,
                     path=args.model.pretrained_path, 
                     smooth=args.model.smooth, fuse=args.model.smooth).to(device)
    
print("Maximum DIM: ",find_max_dim(model))

val_loss, acc = validate_single(model, val_loader, nn.CrossEntropyLoss(), args=args, device=device)
print(f"Initial Permutated model Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc*100:.2f}%")
    
checkpoint = model.learnable_parameter
number_param = len(checkpoint)
print(f"Number of parameters to be learned: {number_param}")
print(f"Parameters keys: {model.keys}")
        
os.makedirs(args.training.save_model_path, exist_ok=True)

start_block = 1
start_epoch = 0
best_acc = 0.0
end_epoch = args.experiment.num_epochs + 1 if not args.model.single_block else args.experiment.num_epochs // 4 + 1
backbone_parameters = {}
print(f"Resuming from checkpoint: {args.resume_from}")
        
trained_blocks = load_trained_blocks(args.resume_from)
dim_dict, gt_model_dict = init_model_dict(args, trained_blocks, args.model.single_block)
dim_dict = shuffle_coordiates_all(dim_dict)
_, _, _, keys_list, _, _, _ = dim_dict[f"{256}"]
selected_keys = np.unique(keys_list)
hyper_model = get_hypernet(args, 4 * trained_blocks, total_param=number_param ,key_list=selected_keys,device=device)

if args.hyper_model.get('use_ema', True):
    ema = EMA(hyper_model, decay=args.hyper_model.ema_decay)
else:
    ema = None
criterion, val_criterion, optimizer, scheduler = get_optimizer(args, hyper_model, first_block=trained_blocks==1) 

checkpoint_info, hyper_model, optimizer, scheduler, ema = load_checkpoint(args.resume_from, hyper_model, optimizer, scheduler, ema, args=args)

if optimizer is None:
    criterion, val_criterion, optimizer, scheduler = get_optimizer(args, hyper_model, first_block=trained_blocks==1)

start_epoch = checkpoint_info['epoch']
best_acc = checkpoint_info['best_acc']
start_block = 7
backbone_parameters = checkpoint_info['backbone_parameters']
print(f"Resuming from block: {start_block}, epoch: {start_epoch}, best accuracy: {best_acc*100:.2f}%")
# Note: If there are more elements to retrieve, do so here.  
del checkpoint_info
gc.collect()

if ema:
    ema.apply()
    del ema
gc.collect()
torch.cuda.empty_cache()
dims = [32, 64, 128, 256]
for dim in dims:
    model = create_model(args.model.type, 
                            hidden_dim=dim,
                            num_param=7,
                            bottom_up=args.model.bottom_up,
                            single_block=False,
                            path=args.model.pretrained_path, 
                            smooth=args.model.smooth, fuse=args.model.fuse,
                            prior=False)   
    if device=="cuda" and torch.backends.cudnn.version() >= 7603:
        model = model.to(device, memory_format=torch.channels_last)  # Module parameters need to be channels last
    else:
        model = model.to(device)

    sampled_model = sample_merge_model(hyper_model, model, args, backbone_parameters=backbone_parameters ,device=device)
    train_loss, train_acc = validate_single(sampled_model, train_loader, val_criterion, args=args, device=device)
    val_loss, val_acc = validate_single(sampled_model, val_loader, val_criterion, args=args, device=device)
    print(f"Validation Loss for dim {dim}: {val_loss:.4f}, Validation Accuracy: {val_acc*100:.2f}%")
    print(f"Validation Loss for dim {dim}: {train_loss:.4f}, Validation Accuracy: {train_acc*100:.2f}%")
    del model
    del sampled_model
    gc.collect()
    torch.cuda.empty_cache()
    