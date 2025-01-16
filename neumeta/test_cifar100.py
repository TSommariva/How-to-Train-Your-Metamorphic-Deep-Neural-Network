import os
import random
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
                           set_seed, shuffle_coordiates_all, #validate, validate_merge, 
                           validate_single, 
                           initialize_wandb,find_max_dim, register_hooks_and_print_shapes, extend_nerf_compose, load_trained_blocks)

import wandb
from sklearn.metrics import accuracy_score

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def get_num_workers():
    try:
        return int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    except (ValueError, TypeError):
        return 1

   
def main_test_nerf(args):

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
    
    
    
    if not args.resume_from :
        print("test requirest resume_from argument")
        return -1
    
    print(f"Resuming from checkpoint: {args.resume_from}")
    
    hyper_model = get_hypernet(args, 4 * (load_trained_blocks(args.resume_from) ),key_list=model.keys ,device=device)
    ema = EMA(hyper_model, decay=args.hyper_model.ema_decay)

    criterion, val_criterion, optimizer, scheduler = get_optimizer(args, hyper_model) 
    
    checkpoint_info, hyper_model, optimizer, scheduler, ema = load_checkpoint(args.resume_from, hyper_model, optimizer, scheduler, ema,args=args)
    
    filename = f"cifar100_results_{args.experiment.name}.txt"
    filepath = os.path.join(args.training.save_model_path, filename)
    
    #ema = False 
    accuracies = []
    if ema:
            print("Applying EMA")
            ema.apply()
    for hidden_dim in range(args.dimensions.test_range[0], args.dimensions.test_range[1] + 1):
        model = create_model(args.model.type, 
                                hidden_dim=hidden_dim,
                                path=args.model.pretrained_path,
                                num_param=args.model.num_param, 
                                bottom_up=args.model.bottom_up,
                                single_block=False,
                                smooth=args.model.smooth, fuse=args.model.fuse).to(device)
                    # Apply Exponential Moving Average (EMA) if enabled
        
            
        accumulated_model = sample_merge_model(hyper_model, model, args, K=100, device=device)
        val_loss, acc = validate_single(accumulated_model, val_loader, val_criterion, args=args, device=device)
        
        accuracies.append(acc)
        print(f"Test using model {args.model}: hidden_dim {hidden_dim}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc*100:.2f}%")
        
        
        # Write the results. 'a' is used to append the results; a new file will be created if it doesn't exist.
        with open(filepath, "a") as file:
            file.write(f"Hidden_dim: {hidden_dim}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc*100:.2f}%\n")
        
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(f"Mean Validation Accuracy: {mean_accuracy * 100:.2f}% ± {std_accuracy * 100:.2f}%")
    
if __name__ == "__main__":
    args = parse_args()
    print_omegaconf(args)
    
    main_test_nerf(args)