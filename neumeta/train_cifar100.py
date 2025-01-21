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
                           validate_single, 
                           initialize_wandb,find_max_dim, register_hooks_and_print_shapes, extend_nerf_compose, load_trained_blocks)

import wandb
from sklearn.metrics import accuracy_score

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
    for dim in range(args.dimensions.range[0], args.dimensions.range[1] + 1):
        model_cls = create_model(args.model.type, 
                                    hidden_dim=dim, num_param=num_blocks, bottom_up=args.model.bottom_up,single_block=single_block ,
                                    path=args.model.pretrained_path, smooth=args.model.smooth, fuse=args.model.smooth, prior=False)
         
        if device=="cuda" and torch.backends.cudnn.version() >= 7603:
            model_cls = model_cls.to(device, memory_format=torch.channels_last)  # Module parameters need to be channels last
        else:
            model_cls = model_cls.to(device)
            
        coords_tensor, keys_list, indices_list, size_list = sample_coordinates(model_cls)
        dim_dict[f"{dim}"] = (model_cls, coords_tensor, keys_list, indices_list, size_list, None)
        
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
                model_trained = model_trained.to(device, memory_format=torch.channels_last)  # Module parameters need to be channels last
            else:
                model_trained = model_trained.to(device)
            model_trained.eval()
            
            gt_model_dict[f"{dim}"] = model_trained
    return dim_dict, gt_model_dict

def train_one_epoch(model, train_loader, optimizer, criterion, dim_dict, gt_model_dict, epoch_idx ,ema=None, args=None, block_idx=1, max_epochs=200):
    model.train()
    optimizer.zero_grad()
    
    no_accumulation = (args.experiment.arch_accumulation_steps * args.experiment.batch_accumulation_steps) == 1
    step = 0
    
    losses = AverageMeter()
    cls_losses = AverageMeter()
    reg_losses = AverageMeter()
    reconstruct_losses = AverageMeter()
    accuracies = AverageMeter()
    extracted_dim = []
    
    for batch_idx, (x, target) in enumerate(train_loader):
        if device=="cuda" and torch.backends.cudnn.version() >= 7603:
            x, target = x.to(device, memory_format=torch.channels_last), target.to(device)
        else:
            x, target = x.to(device), target.to(device)
        
        step +=1
        if (step != 1) or no_accumulation:
            hidden_dim = random.choice(range(args.dimensions.range[0], args.dimensions.range[1] + 1))
            #if hidden dim has already been extracted for this accumulation step, extract another one
            #if not hidden_dim in extracted_dim:
            #    extracted_dim.append(hidden_dim)
            #else:
            #    while hidden_dim in extracted_dim:
            #        hidden_dim = random.choice(range(args.dimensions.range[0], args.dimensions.range[1] + 1))
            #    extracted_dim.append(hidden_dim)
        else:
            hidden_dim = 64
        #    extracted_dim.append(hidden_dim)
            
        model_cls, coords_tensor, keys_list, indices_list, size_list, key_mask = dim_dict[f"{hidden_dim}"]
        selected_keys = np.unique(keys_list)
        #coords_tensor, keys_list, indices_list, size_list, selected_keys = sample_subset(coords_tensor, keys_list, indices_list, size_list, key_mask, ratio=args.ratio)
        
        #add coordinate noise
        #coords_tensor = coords_tensor + ((torch.rand_like(coords_tensor) - 0.5) * args.training.coordinate_noise)#.clamp(-0.49, 0.49)
        
        model_cls, reconstructed_weights = sample_weights(model, model_cls,
                                                          coords_tensor, keys_list, indices_list, size_list, key_mask, selected_keys,
                                                          device=device, NORM=args.dimensions.norm)
        # Forward pass
        predict = model_cls(x)
        results=torch.argmax(predict,dim=1)
        train_acc=accuracy_score(results.cpu(), target.cpu())
        accuracies.update(train_acc)
        
        # Compute loss
        cls_loss = criterion(predict, target)
        cls_losses.update(cls_loss.item())
        
        # Compute regularization loss
        reg_loss = sum([torch.norm(w, p=2) for w in reconstructed_weights])
        reg_losses.update(reg_loss.item())
        
        # Compute MSE loss
        if f"{hidden_dim}" in gt_model_dict:
            gt_model = gt_model_dict[f"{hidden_dim}"]
            gt_selected_weights = [
                w for k, w in gt_model.learnable_parameter.items() if k in selected_keys]
            reconstruct_loss = torch.mean(torch.stack([F.mse_loss(
                w, w_gt) for w, w_gt in zip(reconstructed_weights, gt_selected_weights)]))
        else:
            reconstruct_loss = torch.tensor(0.0)
        reconstruct_losses.update(reconstruct_loss.item())
        
        ce_weight = args.hyper_model.loss_weight.ce_weight
        reg_weight =  args.hyper_model.loss_weight.reg_weight
        recon_weight = args.hyper_model.loss_weight.recon_weight
        
        loss = ce_weight * cls_loss + reg_weight * reg_loss + recon_weight * reconstruct_loss
        losses.update(loss.item())
        
        # Zero model_cls grads
        for updated_weight in model_cls.parameters():
            updated_weight.grad = None
        
        # Scale loss and do backward pass
        scaled_loss = loss / (args.experiment.arch_accumulation_steps * args.experiment.batch_accumulation_steps)
        scaled_loss.backward(retain_graph=True)
        torch.autograd.backward(reconstructed_weights, [
                        w.grad for k, w in model_cls.named_parameters() if k in selected_keys])
                    
        if batch_idx % args.experiment.log_interval == 0 and not args.experiment.debug:
            wandb.log({
                "Running training accuracy argmax" : accuracies.avg,
                "Running average training loss": losses.avg,
                "Cls Loss": cls_losses.avg,
                "Reg Loss": reg_losses.avg,
                "Reconstruct Loss": reconstruct_losses.avg,
                "Learning rate": optimizer.param_groups[0]['lr']
                })#, step=batch_idx + (epoch_idx - 1) * len(train_loader) + (block_idx - 1) * max_epochs * len(train_loader))
        
        if batch_idx % args.experiment.log_interval == 0:
            print(f"Iteration {batch_idx}: Loss = {losses.avg:.4f}, Reg Loss = {reg_losses.avg:.4f}, Reconstruct Loss = {reconstruct_losses.avg:.4f}, Cls Loss = {cls_losses.avg:.4f}, Accuracy = {accuracies.avg*100:.2f}, Learning rate = {optimizer.param_groups[0]['lr']:.4e}")
            
        if (batch_idx % args.experiment.batch_accumulation_steps) == 0 :
            if args.training.get('clip_grad', 0.0) > 0:
                torch.nn.utils.clip_grad_value_(
                    model.parameters(), args.training.clip_grad)
                
            optimizer.step()        
            optimizer.zero_grad()
            step = 0
            extracted_dim = []
            
            if ema:
                ema.update()  # Update the EMA after each training step
    
    #if step > 0:
    #    optimizer.step()        
    #    if ema:
    #        ema.update()    
    #tr_loss, tr_acc = validate_single(model_cls, train_loader, nn.CrossEntropyLoss(), args=args, device=device)
    if not args.experiment.debug:
        wandb.log({
                    "trainLoss_last model of the epoch" : losses.avg,
                    "trainAcc_last model of the epoch" : accuracies.avg
                })
    return losses.avg, accuracies.avg

def main_nerf(args):
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
    
    #hyper_model = get_hypernet(args, 4, key_list=model.keys[0:4] ,device=device)
    #
    #if args.hyper_model.get('use_ema', True):
    #    ema = EMA(hyper_model, decay=args.hyper_model.ema_decay)
    #else:
    #    ema = None
    #    
    #criterion, val_criterion, optimizer, scheduler = get_optimizer(args, hyper_model, first_block=True) 
    
    start_epoch = 0
    best_acc = 0.0
    end_epoch = args.experiment.num_epochs + 1 if not args.model.single_block else args.experiment.num_epochs // 4 + 1
    
    os.makedirs(args.training.save_model_path, exist_ok=True)

    # If specified, load the checkpoint
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        
        dim_dict, gt_model_dict = init_model_dict(args, args.model.num_param, args.model.single_block)
        dim_dict = shuffle_coordiates_all(dim_dict)
        _, _, keys_list, _, _, _ = dim_dict[f"{64}"]
        selected_keys = np.unique(keys_list)
        hyper_model = get_hypernet(args, number_param=number_param, total_param=number_param ,key_list=selected_keys,device=device)
        
        if args.hyper_model.get('use_ema', True):
            ema = EMA(hyper_model, decay=args.hyper_model.ema_decay)
        else:
            ema = None
    
        criterion, val_criterion, optimizer, scheduler = get_optimizer(args, hyper_model, first_block=True) 
        
        checkpoint_info, hyper_model, optimizer, scheduler, ema = load_checkpoint(args.resume_from, hyper_model, optimizer, scheduler, ema, args=args)
        
        if optimizer is None:
            criterion, val_criterion, optimizer, scheduler = get_optimizer(args, hyper_model, first_block=True)
        
        start_epoch = checkpoint_info['epoch']
        best_acc = checkpoint_info['best_acc']
        print(f"Resuming from epoch: {start_epoch}, best accuracy: {best_acc*100:.2f}%")
        # Note: If there are more elements to retrieve, do so here.
    
    if not args.experiment.debug:
        initialize_wandb(args)
        
    if not args.resume_from:
        dim_dict, gt_model_dict = init_model_dict(args, num_blocks=args.model.num_param, single_block=args.model.single_block)
        dim_dict = shuffle_coordiates_all(dim_dict)
        _, _, keys_list, _, _, _ = dim_dict[f"{64}"]
        selected_keys = np.unique(keys_list)
        hyper_model = get_hypernet(args, number_param=number_param,total_param=number_param ,key_list=selected_keys ,device=device)
        criterion, val_criterion, optimizer, scheduler = get_optimizer(args, hyper_model, first_block=True)  
        if args.hyper_model.get('use_ema', True):
            ema = EMA(hyper_model, decay=args.hyper_model.ema_decay)
        else:
            ema = None
    
    for epoch in range(start_epoch + 1, args.experiment.num_epochs):
        train_loss, train_acc = train_one_epoch(hyper_model, train_loader, optimizer, criterion, dim_dict, gt_model_dict, epoch_idx=epoch, ema=ema, args=args)
        scheduler.step()
        print(f"Epoch[{epoch}/{end_epoch-1}], Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc*100:.2f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        if (epoch % args.experiment.eval_interval) == 0 and epoch != 0:
            if ema:
                ema.apply()
            
            sampled_model = sample_merge_model(hyper_model, dim_dict[f"{args.dimensions.start}"][0], args, device=device)
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
            print(f"Epoch [{epoch}/{args.experiment.num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc*100:.2f}%")
            print(f"Epoch [{epoch}/{args.experiment.num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc*100:.2f}%")
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            
            # Save the checkpoint
            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(f"{args.training.save_model_path}/cifar100_nerf_best.pth",hyper_model,optimizer,scheduler,ema,epoch,best_acc)
                print("------------------------------------------------------------------------------------------------------------------------------")
                print(f"Checkpoint saved at epoch {epoch} with accuracy: {best_acc*100:.2f}%")
                print("------------------------------------------------------------------------------------------------------------------------------")
        
    sampled_model = sample_merge_model(hyper_model, dim_dict[f"{args.dimensions.start}"][0], args, device=device)
    val_loss, val_acc = validate_single(sampled_model, val_loader, val_criterion, args=args, device=device)
    
    save_checkpoint(f"{args.training.save_model_path}/cifar100_nerf_last.pth",hyper_model,optimizer,scheduler,ema,args.experiment.num_epochs,val_acc)
    print("------------------------------------------------------------------------------------------------------------------------------")
    print(f"Checkpoint saved at the end of training with accuracy: {val_acc*100:.2f}%")
    print("------------------------------------------------------------------------------------------------------------------------------")
    
    if not args.experiment.debug:
        wandb.finish()
    
    print("Training finished.")
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    #testing the best model
    best_hyper_model = get_hypernet(args, number_param,total_param = number_param ,key_list = model.keys, device=device)
    last_hyper_model = get_hypernet(args, number_param,total_param = number_param ,key_list = model.keys, device=device)
    checkpoint_info, best_hyper_model, _, _, best_ema = load_checkpoint(f"{args.training.save_model_path}/cifar100_nerf_best.pth", best_hyper_model, optimizer,scheduler ,ema, device=device)
    checkpoint_info, last_hyper_model, _, _, last_ema = load_checkpoint(f"{args.training.save_model_path}/cifar100_nerf_last.pth", last_hyper_model, optimizer,scheduler ,ema, device=device)
    best_accuracies = []
    last_accuracies = []
    
    if best_ema:
            best_ema.apply()
    if last_ema:
            last_ema.apply()

    for hidden_dim in range(args.dimensions.test_range[0], args.dimensions.test_range[1] + 1):
        print(f"--------------------------------------------------------HIDDEN DIM {hidden_dim}--------------------------------------------------------")
        # Create a model for the given hidden dimension
        model = create_model(args.model.type, 
                                hidden_dim=hidden_dim,
                                num_param=args.model.num_param,
                                bottom_up=args.model.bottom_up, 
                                single_block=False,
                                path=args.model.pretrained_path, 
                                smooth=args.model.smooth, fuse=args.model.fuse,
                                prior=False).to(device)
            
        accumulated_model_best = sample_merge_model(best_hyper_model, model, args, K=100, device=device)
        best_val_loss, best_val_acc = validate_single(accumulated_model_best, val_loader, val_criterion, args=args, device=device)
        best_accuracies.append(best_val_acc)
        print(f"\tValidation Loss best NeRF: {best_val_loss:.4f}, Validation Accuracy: {best_val_acc*100:.2f}%")
        
        accumulated_model_last = sample_merge_model(last_hyper_model, model, args, K=100, device=device)
        last_val_loss, last_val_acc = validate_single(accumulated_model_last, val_loader, val_criterion, args=args, device=device)
        last_accuracies.append(last_val_acc)
        print(f"\tValidation Loss last NeRF: {last_val_loss:.4f}, Validation Accuracy: {last_val_acc*100:.2f}%")
        
    best_mean_accuracy = np.mean(best_accuracies)
    best_std_accuracy = np.std(best_accuracies)
    last_mean_accuracy = np.mean(last_accuracies)
    lasy_std_accuracy = np.std(last_accuracies)
    
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(f"Best NeRF Mean Validation Accuracy: {best_mean_accuracy * 100:.2f}% ± {best_std_accuracy * 100:.2f}%")
    print(f"Last NeRF Mean Validation Accuracy: {last_mean_accuracy * 100:.2f}% ± {lasy_std_accuracy * 100:.2f}%")
    
    
if __name__ == "__main__":
    args = parse_args()
    print_omegaconf(args)
    
    main_nerf(args)
    