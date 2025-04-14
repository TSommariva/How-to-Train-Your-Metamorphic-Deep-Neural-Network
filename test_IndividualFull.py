import copy
import io
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neumeta.hypermodel import NeRF_MLP_Compose, NeRF_ResMLP_Compose
from neumeta.models import create_model_cifar100 as create_model
from neumeta.utils import (AverageMeter, EMA, create_key_masks, get_cifar100,
                           get_hypernet, get_optimizer, load_checkpoint,
                           parse_args, print_omegaconf, sample_coordinates, sample_weights, sample_merge_model,
                           sample_subset,  save_checkpoint,
                           set_seed, shuffle_coordiates_all,
                           validate_single, validate_all_dimensions,
                           initialize_wandb,find_max_dim, register_hooks_and_print_shapes, extend_nerf_compose, load_trained_blocks,
                           get_cifar_optimizer, weight_difference)

import wandb
import gc
import time
import contextlib
from torchsummary import summary

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
def init_model_dict(args, num_blocks = 1, single_block = False,first_meta_layer=3,num_layers=3):
    """
    Initializes a dictionary of models for each dimension in the given range, along with ground truth models for the starting dimension.

    Args:
        args: An object containing the arguments for initializing the models.

    Returns:
        dim_dict: A dictionary containing the models for each dimension, along with their corresponding coordinates, keys, indices, size, and ground truth models.
        gt_model_dict: A dictionary containing the ground truth models for the starting dimension.
    """
    print(f"INITIALIZE DICTIONARY OF MODELS FOR BLOCK {num_blocks}")
    dim_dict = {}
    gt_model_dict = {}
    if not args.experiment.iterative:
        num_blocks=args.model.num_param
    for dim in range(args.dimensions.range[0], args.dimensions.range[1] + 1):
        model_cls = create_model(args.model.type, 
                             hidden_dim=dim, num_param=num_blocks, bottom_up=args.model.bottom_up,single_block=single_block ,
                             path=args.model.pretrained_path, smooth=args.model.smooth, fuse=args.model.smooth, prior=False, config_args=args, first_meta_layer=first_meta_layer, num_layers=num_layers)
         
        if device=="cuda" and torch.backends.cudnn.version() >= 7603:
            model_cls.to(device, memory_format=torch.channels_last)  # Module parameters need to be channels last
        else:
            model_cls.to(device)
        
        optimizer = get_cifar_optimizer(args, model_cls, num_blocks,num_layers)
            
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
            with contextlib.redirect_stdout(io.StringIO()):
                model_trained = create_model(args.model.type, 
                                 hidden_dim=dim, num_param=num_blocks, bottom_up=args.model.bottom_up, single_block=single_block,
                                 path=args.model.pretrained_path, smooth=args.model.smooth, fuse=args.model.smooth, config_args=args, first_meta_layer=first_meta_layer, num_layers=num_layers)
            
            if device=="cuda" and torch.backends.cudnn.version() >= 7603:
                model_trained.to(device, memory_format=torch.channels_last)  # Module parameters need to be channels last
            else:
                model_trained.to(device)
            model_trained.eval()
            
            gt_model_dict[f"{dim}"] = model_trained
    return dim_dict, gt_model_dict

args = parse_args()
print_omegaconf(args)

model1 = create_model(args.model.type, 
                         hidden_dim=args.dimensions.start,
                         num_param=args.model.num_param,
                         bottom_up=args.model.bottom_up,
                         path=args.model.pretrained_path, 
                         smooth=args.model.smooth, fuse=args.model.smooth,
                         config_args=args).to(device)

test_path = ''
checkpoint_info, best_hyper_model, _, _, _ = load_checkpoint(f"{test_path}/layer1/nerf_block8_best.pth", best_hyper_model, None,None ,None, device=device)
checkpoint_info, best_hyper_model, _, _, _ = load_checkpoint(f"{test_path}/layer2/nerf_block8_best.pth", best_hyper_model, None,None ,None, device=device)
checkpoint_info, best_hyper_model, _, _, _ = load_checkpoint(f"{test_path}/layer3/nerf_block7_best.pth", best_hyper_model, None,None ,None, device=device)