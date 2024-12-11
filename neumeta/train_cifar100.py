import argparse
import copy
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neumeta.hypermodel import NeRF_MLP_Compose, NeRF_ResMLP_Compose
from neumeta.models import create_model_cifar100 as create_model
from neumeta.models import create_model_cifar100_slim
from neumeta.utils import (AverageMeter, EMA, create_key_masks, get_cifar100,
                           get_hypernet, get_optimizer, load_checkpoint,
                           parse_args, print_omegaconf, sample_coordinates,
                           sample_subset, sample_weights, save_checkpoint,
                           set_seed, shuffle_coordiates_all, #validate, validate_merge, 
                           validate_single, sample_merge_model)
import wandb
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from neumeta.models import BasicBlock, BasicBlock_Resize

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def find_max_dim(model_cls):
    checkpoint = model_cls.learnable_parameter
    
    max_value = len(checkpoint)
    # Iterate over the new model's weights
    for i, (k, tensor) in enumerate(checkpoint.items()):
        
        # Handle 2D tensors (e.g., weight matrices)
        if len(tensor.shape) == 4:
            coords = [tensor.shape[0], tensor.shape[1]]
            max_value = max(max_value, max(coords))
                    
        elif len(tensor.shape) == 2:

            coords = [tensor.shape[0], tensor.shape[1]]
            max_value = max(max_value, max(coords))
                    
        # Handle 1D tensors (e.g., biases)
        elif len(tensor.shape) == 1:
          
            max_value = max(max_value, tensor.shape[0])
    
    return max_value
    
def initialize_wandb(config):
    import time
    """
    Initializes Weights and Biases (wandb) with the given configuration.
    
    Args:
        configuration (dict): Configuration parameters for the run.
    """
    # Name the run using current time and configuration name
    run_name = f"{config.experiment.name}-{time.strftime('%Y%m%d%H%M%S')}"
    
    wandb.init(project="ninr", name=run_name, config=dict(config), group='cifar100')

    

def train_one_epoch(model, train_loader, optimizer, criterion, dim_dict, gt_model_dict, epoch_idx, ema=None, args=None):
    model.train()
    total_loss = 0.0

    losses = AverageMeter()
    cls_losses = AverageMeter()
    reg_losses = AverageMeter()
    reconstruct_losses = AverageMeter()

    for batch_idx, (x, target) in enumerate(train_loader):
        optimizer.zero_grad()
        x, target = x.to(device), target.to(device)
        hidden_dim = random.choice(args.dimensions.range)
        model_cls, coords_tensor, keys_list, indices_list, size_list, key_mask = dim_dict[f"{hidden_dim}"]
        coords_tensor, keys_list, indices_list, size_list, selected_keys = sample_subset(coords_tensor,
                                                                                         keys_list,
                                                                                         indices_list,
                                                                                         size_list,
                                                                                         key_mask,
                                                                                         ratio=args.ratio)
        if args.training.coordinate_noise > 0.0:
            coords_tensor = coords_tensor + (torch.rand_like(coords_tensor) - 0.5) * args.training.coordinate_noise
        model_cls, reconstructed_weights = sample_weights(model, model_cls,
                                                          coords_tensor, keys_list, indices_list, size_list, key_mask, selected_keys,
                                                          device=device, NORM=args.dimensions.norm)

        # Forward pass
        predict = model_cls(x)
        
        results=torch.argmax(predict,dim=1)
        train_acc=accuracy_score(results.cpu(), target.cpu())
        
        # Compute loss
        cls_loss = criterion(predict, target)  # * 0.01
        # Compute regularization loss
        reg_loss = sum([torch.norm(w, p=2)
                                for w in reconstructed_weights])

        if f"{hidden_dim}" in gt_model_dict:
            gt_model = gt_model_dict[f"{hidden_dim}"]
            gt_selected_weights = [
                w for k, w in gt_model.learnable_parameter.items() if k in selected_keys]

            reconstruct_loss = torch.mean(torch.stack([F.mse_loss(
                w, w_gt) for w, w_gt in zip(reconstructed_weights, gt_selected_weights)]))
        else:
            reconstruct_loss = torch.tensor(0.0)

        loss = args.hyper_model.loss_weight.ce_weight * cls_loss + args.hyper_model.loss_weight.reg_weight * \
            reg_loss + args.hyper_model.loss_weight.recon_weight * reconstruct_loss

        for updated_weight in model_cls.parameters():
            updated_weight.grad = None

        loss.backward(retain_graph=True)
        torch.autograd.backward(reconstructed_weights, [
                                w.grad for k, w in model_cls.named_parameters() if k in selected_keys])

        if args.training.get('clip_grad', 0.0) > 0:
            torch.nn.utils.clip_grad_value_(
                model.parameters(), args.training.clip_grad)

        optimizer.step()
        if ema:
            ema.update()  # Update the EMA after each training step
        total_loss += loss.item()

        losses.update(loss.item())
        cls_losses.update(cls_loss.item())
        reg_losses.update(reg_loss.item())
        reconstruct_losses.update(reconstruct_loss.item())

        if batch_idx % args.experiment.log_interval == 0 and not args.experiment.debug:
            wandb.log({
                "Running training accuracy argmax" : train_acc,
                "Running average training loss": losses.avg,
                "Cls Loss": cls_losses.avg,
                "Reg Loss": reg_losses.avg,
                "Reconstruct Loss": reconstruct_losses.avg,
                "Learning rate": optimizer.param_groups[0]['lr']
            }, step=batch_idx + epoch_idx * len(train_loader))
            print(
                f"Iteration {batch_idx}: Loss = {losses.avg:.4f}, Reg Loss = {reg_losses.avg:.4f}, Reconstruct Loss = {reconstruct_losses.avg:.4f}, Cls Loss = {cls_losses.avg:.4f}, Learning rate = {optimizer.param_groups[0]['lr']:.4e}")
    
    tr_loss, tr_acc = validate_single(model_cls, train_loader, nn.CrossEntropyLoss(), args=args, device=device)
    
    if not args.experiment.debug:
        wandb.log({
                    "trainLoss_last model of the epoch" : tr_loss,
                    "trainAcc_last model of the epoch" : tr_acc

                })
    return losses.avg, dim_dict, gt_model_dict

def register_hooks_and_print_shapes(model, input_tensor):
    output_shapes = {}
    learnable_keys = set(model.learnable_parameter.keys())
    
    def hook_fnc(module_name):
        def hook_fn(module, input, output):
            class_name = module.__class__.__name__
            #module_idx = len(output_shapes)
            m_key = f"{module_name}_{class_name}"
            output_shapes[m_key] = output.shape
        return hook_fn

    # Register hooks to all layers
    hooks = []
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Sequential, nn.ModuleList, BasicBlock, BasicBlock_Resize)) and module != model:
            if any(key.startswith(name) for key in learnable_keys):
                hook = module.register_forward_hook(hook_fnc(name))
                hooks.append(hook)

    # Perform a forward pass to trigger the hooks
    model(input_tensor)

    # Print the output shapes
    for key, shape in output_shapes.items():
        print(f"{key}: {shape}")

    # Remove hooks after use
    for hook in hooks:
        hook.remove()

def init_model_dict(args):
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
    for dim in args.dimensions.range:
        # for dp in depth_range:
        model_cls = create_model(args.model.type, hidden_dim=dim,num_param=args.model.num_param, path=args.model.pretrained_path, smooth=args.model.smooth, fuse=args.model.smooth).to(device)
        #model_cls = create_model_cifar100_slim (args.model.type, hidden_dim=dim, args.model.num_param, path=args.model.pretrained_path, smooth=args.model.smooth, fuse=args.model.smooth).to(device)
        
        coords_tensor, keys_list, indices_list, size_list = sample_coordinates(model_cls)
        dim_dict[f"{dim}"] = (model_cls, coords_tensor, keys_list, indices_list, size_list, None)
        
        #input_tensor = torch.randn(1, 3, 32, 32).to(device)
        #register_hooks_and_print_shapes(model_cls, input_tensor)

        if dim == args.dimensions.start:
            print(f"Loading model for dim {dim}")
            model_trained = create_model(args.model.type, hidden_dim=dim,num_param=args.model.num_param, path=args.model.pretrained_path, smooth=args.model.smooth, fuse=args.model.smooth).to(device)
            #model_trained = create_model_cifar100_slim(args.model.type, hidden_dim=dim,num_param=args.model.num_param ,path=args.model.pretrained_path, smooth=args.model.smooth, fuse=args.model.smooth).to(device)
            model_trained.eval()
            
            gt_model_dict[f"{dim}"] = model_trained
    return dim_dict, gt_model_dict

def main_nerf():
    args = parse_args()
    print_omegaconf(args)
    
    #if args.model.num_param < 3:
    #    raise ValueError(f"num_param must be >= 3, got {args.model.num_param}")

    set_seed(args.experiment.seed)
    train_loader, val_loader = get_cifar100(args.training.batch_size, 
                                           strong_transform=args.training.get('strong_aug', None),
                                           )
    
    # checkpoint = torch.load(path, map_location='cpu')
    #model = create_model_cifar100_slim(args.model.type, 
    #                     hidden_dim=args.dimensions.start,
    #                     num_param=args.model.num_param, 
    #                     path=args.model.pretrained_path, 
    #                     smooth=args.model.smooth, fuse=args.model.smooth).to(device)
    model = create_model(args.model.type, 
                         hidden_dim=args.dimensions.start,
                         num_param=args.model.num_param, 
                         path=args.model.pretrained_path, 
                         smooth=args.model.smooth, fuse=args.model.smooth).to(device)
    
    print("Maximum DIM: ",find_max_dim(model))

    val_loss, acc = validate_single(model, val_loader, nn.CrossEntropyLoss(), args=args)
    print(f"Initial Permutated model Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc*100:.2f}%")
    checkpoint = model.learnable_parameter
    # print(checkpoint)
    number_param = len(checkpoint)
    print(f"Number of parameters to be learned: {number_param}")
    print(f"Parameters keys: {model.keys}")
    hyper_model = get_hypernet(args, number_param)
    ema = EMA(hyper_model, decay=args.hyper_model.ema_decay)
    criterion, val_criterion, optimizer, scheduler = get_optimizer(args, hyper_model)
    
    start_epoch = 0
    best_acc = 0.0
    
    os.makedirs(args.training.save_model_path, exist_ok=True)

    # If specified, load the checkpoint
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint_info, hyper_model = load_checkpoint(args.resume_from, hyper_model, optimizer, scheduler, ema)
        start_epoch = checkpoint_info['epoch']
        best_acc = checkpoint_info['best_acc']
        print(f"Resuming from epoch: {start_epoch}, best accuracy: {best_acc*100:.2f}%")
        # Note: If there are more elements to retrieve, do so here.
    
    if args.test == False:
        if not args.experiment.debug:
            initialize_wandb(args)
        dim_dict, gt_model_dict = init_model_dict(args)
        dim_dict = shuffle_coordiates_all(dim_dict)
        
        for epoch in range(start_epoch, args.experiment.num_epochs):
            train_loss, dim_dict, gt_model_dict = train_one_epoch(hyper_model, train_loader, optimizer, criterion, dim_dict, gt_model_dict, epoch_idx=epoch, ema=ema, args=args)
            scheduler.step()

            print(f"Epoch [{epoch+1}/{args.experiment.num_epochs}], Training Loss: {train_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

            if (epoch + 1) % 1 == 0: #args.experiment.eval_interval == 0:
                if ema:
                    ema.apply()

                sampled_model = sample_merge_model(hyper_model, gt_model_dict[f"{args.dimensions.start}"], args, device=device)
                train_loss, train_acc = validate_single(sampled_model, train_loader, val_criterion, args=args, device=device)
                val_loss, val_acc = validate_single(sampled_model, val_loader, val_criterion, args=args, device=device)
                
                if ema:
                    ema.restore()
                if not args.experiment.debug:    
                    wandb.log({
                        "Train Loss_model sampled outside training": train_loss,
                        "Train Accuracy_model sampled outside training": train_acc,
                        "Validation Loss_model sampled outside training": val_loss,
                        "Validation Accuracy_model sampled outside training": val_acc
                    })
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print(f"Epoch [{epoch+1}/{args.experiment.num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc*100:.2f}%")
                print(f"Epoch [{epoch+1}/{args.experiment.num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc*100:.2f}%")
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                
                # Save the checkpoint
                if val_acc > best_acc:
                    best_acc = val_acc
                    save_checkpoint(f"{args.training.save_model_path}/cifar100_nerf_best.pth",hyper_model,optimizer,scheduler,ema,epoch,best_acc)
                    print("------------------------------------------------------------------------------------------------------------------------------")
                    print(f"Checkpoint saved at epoch {epoch} with accuracy: {best_acc*100:.2f}%")
                    print("------------------------------------------------------------------------------------------------------------------------------")
        if not args.experiment.debug:
            wandb.finish()
        
        print("Training finished.")
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        #testing the best model
        checkpoint_info, hyper_model = load_checkpoint(f"{args.training.save_model_path}/cifar100_nerf_best.pth", hyper_model, optimizer,scheduler ,ema, device=device)
        accuracies = []
        for hidden_dim in range(16, 65):
            # Create a model for the given hidden dimension
            model = create_model(args.model.type, 
                                    hidden_dim=hidden_dim,
                                    num_param=args.model.num_param,
                                    path=args.model.pretrained_path, 
                                    smooth=args.model.smooth, fuse=args.model.fuse).to(device)
            #model = create_model_cifar100_slim(args.model.type, 
            #                        hidden_dim=hidden_dim,
            #                        num_param=args.model.num_param, 
            #                        path=args.model.pretrained_path, 
            #                        smooth=args.model.smooth, fuse=args.model.fuse).to(device)

            # Sample the merged model for K times
            accumulated_model = sample_merge_model(hyper_model, model, args, K=100, device=device)

            # Validate the merged model
            val_loss, val_acc = validate_single(accumulated_model, val_loader, val_criterion, args=args, device=device)
            accuracies.append(val_acc)

            # Print the results
            print(f"Test using model {args.model}: hidden_dim {hidden_dim}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc*100:.2f}%")
        
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print(f"Mean Validation Accuracy: {mean_accuracy * 100:.2f}% ± {std_accuracy * 100:.2f}%")
    else:
        accuracies = []
        for hidden_dim in range(16, 65):
            model = create_model(args.model.type, 
                                    hidden_dim=hidden_dim,
                                    path=args.model.pretrained_path,
                                    num_param=args.model.num_param, 
                                    smooth=args.model.smooth, fuse=args.model.fuse).to(device)
            #model = create_model_cifar100_slim(args.model.type, 
            #                     hidden_dim=hidden_dim,
            #                     num_param=args.model.num_param, 
            #                     path=args.model.pretrained_path, 
            #                     smooth=args.model.smooth, fuse=args.model.smooth).to(device)

            for valid_fn in [validate_single]:
                print(f"Testing using fn {valid_fn.__name__}")

                # Apply Exponential Moving Average (EMA) if enabled
                if ema:
                    print("Applying EMA")
                    ema.apply()
                    val_loss, acc = valid_fn(model, val_loader, val_criterion, args=args, device=device)
                    ema.restore()  # Restore the original weights after applying EMA
                else:
                    val_loss, acc = valid_fn(hyper_model, val_loader, val_criterion, model_cls=model, args=args)
                    
                accuracies.append(acc)

                print(f"Test using model {args.model}: hidden_dim {hidden_dim}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc*100:.2f}%")
                
                # Define the directory and filename structure
                filename = f"cifar100_{valid_fn.__name__}_results_{args.experiment.name}.txt"
                filepath = os.path.join(args.training.save_model_path, filename)


                # Write the results. 'a' is used to append the results; a new file will be created if it doesn't exist.
                with open(filepath, "a") as file:
                    file.write(f"Hidden_dim: {hidden_dim}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc*100:.2f}%\n")
            
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print(f"Mean Validation Accuracy: {mean_accuracy * 100:.2f}% ± {std_accuracy * 100:.2f}%")

    print("Training finished.")

    
if __name__ == "__main__":
    # main_fit_nerf()
    main_nerf()
# 