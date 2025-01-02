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
                                 path=args.model.pretrained_path, smooth=args.model.smooth, fuse=args.model.smooth)
        #model_cls = create_model_cifar100_slim (args.model.type, hidden_dim=dim, num_blocks, path=args.model.pretrained_path, smooth=args.model.smooth, fuse=args.model.smooth).to(device)
         
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
            #model_trained = create_model_cifar100_slim(args.model.type, hidden_dim=dim,num_param=num_blocks ,path=args.model.pretrained_path, smooth=args.model.smooth, fuse=args.model.smooth).to(device)
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
    extracted_dim = []
    
    for batch_idx, (x, target) in enumerate(train_loader):
        if device=="cuda" and torch.backends.cudnn.version() >= 7603:
            x, target = x.to(device, memory_format=torch.channels_last), target.to(device)
        else:
            x, target = x.to(device), target.to(device)
        
        #gradient aggregation
        for arch_step in range(args.experiment.arch_accumulation_steps):
            step +=1
            
            if (step != 1) or no_accumulation:
                hidden_dim = random.choice(range(args.dimensions.range[0], args.dimensions.range[1] + 1))
                
                #if hidden dim has already been extracted for this accumulation step, extract another one
                if not hidden_dim in extracted_dim:
                    extracted_dim.append(hidden_dim)
                else:
                    while hidden_dim in extracted_dim:
                        hidden_dim = random.choice(range(args.dimensions.range[0], args.dimensions.range[1] + 1))
                    extracted_dim.append(hidden_dim)
            else:
                hidden_dim = 64
                extracted_dim.append(hidden_dim)
                
            model_cls, coords_tensor, keys_list, indices_list, size_list, key_mask = dim_dict[f"{hidden_dim}"]
            selected_keys = np.unique(keys_list)
            #coords_tensor, keys_list, indices_list, size_list, selected_keys = sample_subset(coords_tensor, keys_list, indices_list, size_list, key_mask, ratio=args.ratio)
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
            cls_loss = criterion(predict, target)
            cls_losses.update(cls_loss.item())
            # Compute regularization loss
            reg_loss = sum([torch.norm(w, p=2)
                                    for w in reconstructed_weights])
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
            loss = args.hyper_model.loss_weight.ce_weight * cls_loss + args.hyper_model.loss_weight.reg_weight * \
                reg_loss + args.hyper_model.loss_weight.recon_weight * reconstruct_loss
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
                "Running training accuracy argmax" : train_acc,
                "Running average training loss": losses.avg,
                "Cls Loss": cls_losses.avg,
                "Reg Loss": reg_losses.avg,
                "Reconstruct Loss": reconstruct_losses.avg,
                
            }, commit=False)#, step=batch_idx + (epoch_idx - 1) * len(train_loader) + (batch_idx - 1) * max_epochs * len(train_loader))
            for i, paramgroup in enumerate(optimizer.param_groups):
                if i == 0:
                    wandb.log({
                        f"Learning rate": paramgroup['lr']
                })
                else:
                    wandb.log({
                        f"Fine-tuning Learning rate {i}": paramgroup['lr']
                    })
        if batch_idx % args.experiment.log_interval == 0:
            for i, paramgroup in enumerate(optimizer.param_groups):
                if i == 0:
                    print(f"Iteration {batch_idx}: Loss = {losses.avg:.4f}, Reg Loss = {reg_losses.avg:.4f}, Reconstruct Loss = {reconstruct_losses.avg:.4f}, Cls Loss = {cls_losses.avg:.4f}, Learning rate = {paramgroup['lr']:.4e}")
                else:
                    print(f"\tFine-tuning Learning rate = {paramgroup['lr']:.4e}")
            
            losses.reset()
            cls_losses.reset()
            reg_losses.reset()
            reconstruct_losses.reset()
            
            
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
    
    tr_loss, tr_acc = validate_single(model_cls, train_loader, nn.CrossEntropyLoss(), args=args, device=device)
    
    if not args.experiment.debug:
        wandb.log({
                    "trainLoss_last model of the epoch" : tr_loss,
                    "trainAcc_last model of the epoch" : tr_acc

                })
    return tr_loss

def copyParams(NerF_src, Nerf_dest):
    print("initialize MLPs of the current block with the weights of the previous one")
    with torch.no_grad():
        for i in range(4):
            for name, param in NerF_src[i].named_parameters():
                dest_param = dict(Nerf_dest.model[i].named_parameters())[name]
                if dest_param.shape == param.shape:
                    dest_param.copy_(param)
                else:
                    print(f"src:{name} and dest params have different shapes")
    return Nerf_dest

    
def main_iterative_nerf(args):

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
    
    if(number_param %4 != 0):
        print("Only residual blocks with no projection in the skip connection are supported")
        return -1
    
            
    os.makedirs(args.training.save_model_path, exist_ok=True)
    
    hyper_model = get_hypernet(args, 4, device=device)
    ema = EMA(hyper_model, decay=args.hyper_model.ema_decay)
    criterion, val_criterion, optimizer, scheduler = get_optimizer(args, hyper_model) 

    start_block = 1
    start_epoch = 0
    best_acc = 0.0
    end_epoch = args.experiment.num_epochs + 1 if not args.model.single_block else args.experiment.num_epochs // 4 + 1
    
    
    frozen_NeRF = NeRF_ResMLP_Compose(
        input_dim=args.hyper_model.input_dim,
        hidden_dim=args.hyper_model.hidden_dim,
        num_layers=args.hyper_model.num_layers,
        output_dim=args.hyper_model.output_dim,
        num_freqs=args.hyper_model.num_freqs,
        scalar=args.hyper_model.get('scalar', 0.1),
        num_compose=0).to(device)

    # If specified, load the checkpoint
    if args.resume_from and "fineTuning" not in args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        
        #changed
        hyper_model = get_hypernet(args, 4 * (load_trained_blocks(args.resume_from) ), device=device)
        
        if args.model.single_block:
            hyper_model = get_hypernet(args, 4, device=device)
            
        criterion, val_criterion, optimizer, scheduler = get_optimizer(args, hyper_model) 
        
        checkpoint_info, hyper_model = load_checkpoint(args.resume_from, hyper_model, optimizer, scheduler, ema,args=args)
        if optimizer is None:
            criterion, val_criterion, optimizer, scheduler = get_optimizer(args, hyper_model)
        
        start_epoch = checkpoint_info['epoch']
        best_acc = checkpoint_info['best_acc']
        start_block = checkpoint_info['trained_blocks']
        print(f"Resuming from block: {start_block}, epoch: {start_epoch}, best accuracy: {best_acc*100:.2f}%")
        # Note: If there are more elements to retrieve, do so here.
        
    elif args.resume_from and "fineTuning" in args.resume_from:
        start_block = args.model.num_param + 1
        start_epoch = end_epoch
    
    if not args.experiment.debug:
        initialize_wandb(args)
    
    for block_id in range(start_block, args.model.num_param + 1):
        os.makedirs(f"{args.training.save_model_path}/block{block_id}", exist_ok=True)
        print(f"BLOCK[{block_id}/{args.model.num_param}]")
        
        if(block_id != start_block):
            hyper_model = get_hypernet(args, 4, device=device)
            if(args.model.bottom_up):
                if args.experiment.custom_init:
                    hyper_model = copyParams(frozen_NeRF.model[-4:],hyper_model)
                if not args.model.single_block:    
                    hyper_model=extend_nerf_compose(frozen_NeRF,hyper_model)
                    if not args.training.get('ft_scaling', False):
                        criterion, val_criterion, optimizer, scheduler = get_optimizer(args, hyper_model)   
                    else:
                        train_parameters = [p for n, p in hyper_model.model[-4:].named_parameters()]
                        ft_parameters = [p for n, p in hyper_model.model[:-4].named_parameters()]

                        criterion, val_criterion, optimizer, scheduler = get_optimizer_scaledFT(args, hyper_model,train_parameters,ft_parameters)
            else:
                if args.experiment.custom_init:
                    copyParams(frozen_NeRF.model[:4],hyper_model)
                if args.model.single_block:
                    print("top down training with single block approach not supported")
                    return -1
                hyper_model=extend_nerf_compose(hyper_model,frozen_NeRF)
                if not args.training.get('ft_scaling', False):
                    criterion, val_criterion, optimizer, scheduler = get_optimizer(args, hyper_model)   
                else:      
                    train_parameters = [p for n, p in hyper_model.model[:4].named_parameters()]
                    ft_parameters = [p for n, p in hyper_model.model[4:].named_parameters()]
                    criterion, val_criterion, optimizer, scheduler = get_optimizer_scaledFT(args, hyper_model,train_parameters,ft_parameters)

            ema = EMA(hyper_model, decay=args.hyper_model.ema_decay)
            
            start_epoch = 0
            best_acc = 0.0

            
        dim_dict, gt_model_dict = init_model_dict(args, block_id, args.model.single_block)
        dim_dict = shuffle_coordiates_all(dim_dict)
        
        epoch=None
        
        for epoch in range(start_epoch + 1, end_epoch):
            train_loss = train_one_epoch(hyper_model, train_loader, optimizer, criterion, dim_dict, gt_model_dict, epoch_idx=epoch, ema=ema, args=args, block_idx=block_id, max_epochs=end_epoch)
            scheduler.step()

            print(f"Block[{block_id}/{args.model.num_param}]-Epoch[{epoch}/{end_epoch-1}], Training Loss: {train_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

            if epoch % args.experiment.eval_interval == 0:
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
                print(f"Block[{block_id}/{args.model.num_param}]-Epoch[{epoch}/{end_epoch-1}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc*100:.2f}%")
                print(f"Block[{block_id}/{args.model.num_param}]-Epoch[{epoch}/{end_epoch-1}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc*100:.2f}%")
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

                # Save the checkpoint
                if val_acc > best_acc:
                    best_acc = val_acc
                    save_checkpoint(f"{args.training.save_model_path}/block{block_id}/cifar100_nerf_best.pth",hyper_model,optimizer,scheduler,ema,epoch,best_acc, block_id)
                    print("------------------------------------------------------------------------------------------------------------------------------")
                    print(f"Block[{block_id}/{args.model.num_param}] Checkpoint saved at epoch {epoch} with accuracy: {best_acc*100:.2f}%")
                    print("------------------------------------------------------------------------------------------------------------------------------")
        
        if epoch is None:
            epoch = end_epoch
            
        if args.model.single_block:
            save_checkpoint(f"{args.training.save_model_path}/block{block_id}/cifar100_nerf_last.pth",hyper_model,optimizer,scheduler,ema,epoch,best_acc, block_id)
            print("------------------------------------------------------------------------------------------------------------------------------")
            print(f"Block[{block_id}/{args.model.num_param}] Checkpoint saved at the end of block {block_id} with accuracy: {best_acc*100:.2f}%")
            print("------------------------------------------------------------------------------------------------------------------------------")
            
        #for param in hyper_model.parameters():
        #    param.requires_grad = False
        frozen_NeRF = hyper_model
        
                
    if args.model.single_block:
        os.makedirs(f"{args.training.save_model_path}/fineTuning", exist_ok=True)
        hyper_model = NeRF_ResMLP_Compose(
            input_dim=args.hyper_model.input_dim,
            hidden_dim=args.hyper_model.hidden_dim,
            num_layers=args.hyper_model.num_layers,
            output_dim=args.hyper_model.output_dim,
            num_freqs=args.hyper_model.num_freqs,
            scalar=args.hyper_model.get('scalar', 0.1),
            num_compose=0).to(device)
        
        last_hyper_model = NeRF_ResMLP_Compose(
            input_dim=args.hyper_model.input_dim,
            hidden_dim=args.hyper_model.hidden_dim,
            num_layers=args.hyper_model.num_layers,
            output_dim=args.hyper_model.output_dim,
            num_freqs=args.hyper_model.num_freqs,
            scalar=args.hyper_model.get('scalar', 0.1),
            num_compose=4).to(device)
        for block_id in range(1, args.model.num_param + 1):
            checkpoint_info, last_hyper_model = load_checkpoint(f"{args.training.save_model_path}/block{block_id}/cifar100_nerf_last.pth", last_hyper_model, optimizer, scheduler ,ema, device=device)
            hyper_model=extend_nerf_compose(hyper_model,last_hyper_model)
        
        ema = EMA(hyper_model, decay=args.hyper_model.ema_decay)
        criterion, val_criterion, optimizer, scheduler = get_optimizer(args, hyper_model)
        start_epoch = 0
        best_acc = 0.0
        
        if args.resume_from and "fineTuning" in args.resume_from:
            print(f"Resuming from checkpoint: {args.resume_from}")
            
            hyper_model = get_hypernet(args, 4 * (load_trained_blocks(args.resume_from)), device=device)
            criterion, val_criterion, optimizer, scheduler = get_optimizer(args, hyper_model) 
            checkpoint_info, hyper_model = load_checkpoint(args.resume_from, hyper_model, optimizer, scheduler, ema,args=args)
            if optimizer is None:
                criterion, val_criterion, optimizer, scheduler = get_optimizer(args, hyper_model)
                
            start_epoch = checkpoint_info['epoch']
            best_acc = checkpoint_info['best_acc']
            print(f"Resuming from fineTuning, epoch: {start_epoch}, best accuracy: {best_acc*100:.2f}%")
        
        dim_dict, gt_model_dict = init_model_dict(args, args.model.num_param, single_block=False)
        dim_dict = shuffle_coordiates_all(dim_dict)
        
        for epoch in range(start_epoch + 1, args.experiment.num_epochs + 1):
            train_loss = train_one_epoch(hyper_model, train_loader, optimizer, criterion, dim_dict, gt_model_dict, epoch_idx=epoch, ema=ema, args=args, block_idx=args.model.num_param, max_epochs=args.experiment.num_epochs)
            
            scheduler.step()
            print(f"Fine-Tuning - Epoch[{epoch}/{args.experiment.num_epochs}], Training Loss: {train_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            
            if epoch % args.experiment.eval_interval == 0:
                
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
                print(f"Fine-Tuning - Epoch[{epoch}/{args.experiment.num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc*100:.2f}%")
                print(f"Fine-Tuning - Epoch[{epoch}/{args.experiment.num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc*100:.2f}%")
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                
                # Save the checkpoint
                if val_acc > best_acc:
                    best_acc = val_acc
                    save_checkpoint(f"{args.training.save_model_path}/fineTuning/cifar100_nerf_best.pth",hyper_model,optimizer,scheduler,ema,epoch,best_acc, args.model.num_param)
                    print("------------------------------------------------------------------------------------------------------------------------------")
                    print(f"Fine-Tuning Checkpoint saved at epoch {epoch} with accuracy: {best_acc*100:.2f}%")
                    print("------------------------------------------------------------------------------------------------------------------------------")
    
    sampled_model = sample_merge_model(hyper_model, gt_model_dict[f"{args.dimensions.start}"], args, device=device)
    val_loss, val_acc = validate_single(sampled_model, val_loader, val_criterion, args=args, device=device)
    save_checkpoint(f"{args.training.save_model_path}/cifar100_nerf_last.pth",hyper_model,optimizer,scheduler,ema,args.experiment.num_epochs,val_acc, args.model.num_param)
    print("------------------------------------------------------------------------------------------------------------------------------")
    print(f"Checkpoint saved at the end of training with accuracy: {val_acc*100:.2f}%")
    print("------------------------------------------------------------------------------------------------------------------------------")
                    
    if not args.experiment.debug:
        wandb.finish()
    
    print("Training finished.")
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    #testing the best model
    best_hyper_model = get_hypernet(args, number_param, device=device)
    last_hyper_model = get_hypernet(args, number_param, device=device)
    if not args.model.single_block:
        checkpoint_info, best_hyper_model = load_checkpoint(f"{args.training.save_model_path}/block{args.model.num_param}/cifar100_nerf_best.pth", best_hyper_model, optimizer,scheduler ,ema, device=device)
        checkpoint_info, last_hyper_model = load_checkpoint(f"{args.training.save_model_path}/cifar100_nerf_last.pth", last_hyper_model, optimizer,scheduler ,ema, device=device)
    else:
        checkpoint_info, best_hyper_model = load_checkpoint(f"{args.training.save_model_path}/fineTuning/cifar100_nerf_best.pth", best_hyper_model, optimizer,scheduler ,ema, device=device)
        checkpoint_info, last_hyper_model = load_checkpoint(f"{args.training.save_model_path}/cifar100_nerf_last.pth", last_hyper_model, optimizer,scheduler ,ema, device=device)
    
    best_accuracies = []
    last_accuracies = []
    
    for hidden_dim in range(args.dimensions.test_range[0], args.dimensions.test_range[1] + 1):
        # Create a model for the given hidden dimension
        model = create_model(args.model.type, 
                                hidden_dim=hidden_dim,
                                num_param=args.model.num_param,
                                bottom_up=args.model.bottom_up,
                                single_block=False,
                                path=args.model.pretrained_path, 
                                smooth=args.model.smooth, fuse=args.model.fuse).to(device)
        
        # Sample the merged model for K times
        accumulated_model_best = sample_merge_model(best_hyper_model, model, args, K=100, device=device)
        accumulated_model_last = sample_merge_model(last_hyper_model, model, args, K=100, device=device)
        # Validate the merged model
        best_val_loss, best_val_acc = validate_single(accumulated_model_best, val_loader, val_criterion, args=args, device=device)
        last_val_loss, last_val_acc = validate_single(accumulated_model_last, val_loader, val_criterion, args=args, device=device)
        best_accuracies.append(best_val_acc)
        last_accuracies.append(last_val_acc)
        # Print the results
        print(f"Test using model {args.model}: hidden_dim {hidden_dim}")
        print(f"\tValidation Loss best NeRF: {best_val_loss:.4f}, Validation Accuracy: {best_val_acc*100:.2f}%")
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
    
    main_iterative_nerf(args)