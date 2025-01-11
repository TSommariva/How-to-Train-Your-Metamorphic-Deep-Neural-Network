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

def train_one_epoch(weights_hypernet, biases_hypernet, train_loader, w_optimizer, b_optimizer, criterion, dim_dict, gt_model_dict, epoch_idx ,ema=None, args=None, block_idx=1, max_epochs=200):
    weights_hypernet.train()
    biases_hypernet.train()
    w_optimizer.zero_grad() 
    b_optimizer.zero_grad()
    
    no_accumulation = (args.experiment.arch_accumulation_steps * args.experiment.batch_accumulation_steps) == 1
    step = 0
    
    losses = AverageMeter()
    cls_losses = AverageMeter()
    reg_losses = AverageMeter()
    reconstruct_losses = AverageMeter()
    #extracted_dim = []
    
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
            #extracted_dim.append(hidden_dim)
                        
        model_cls, coords_tensor, keys_list, indices_list, size_list, key_mask = dim_dict[f"{hidden_dim}"]
        selected_keys = np.unique(keys_list)
        #coords_tensor, keys_list, indices_list, size_list, selected_keys = sample_subset(coords_tensor, keys_list, indices_list, size_list, key_mask, ratio=args.ratio)
        
        #add coordinate noise
        coords_tensor = coords_tensor + (torch.rand_like(coords_tensor) - 0.5).clamp(-0.49,0.49) * args.training.coordinate_noise
        
        weights_key = [k for k in selected_keys if "weight" in k]
        bias_key = [k for k in selected_keys if "bias" in k]
        
        model_cls, reconstructed_weights_b = sample_weights(biases_hypernet, model_cls,
                                                  coords_tensor, keys_list, indices_list, size_list, key_mask, selected_keys,
                                                  device=device, NORM=args.dimensions.norm, subset_key=bias_key )

        
        model_cls, reconstructed_weights_w = sample_weights(weights_hypernet, model_cls,
                                                          coords_tensor, keys_list, indices_list, size_list, key_mask, selected_keys,
                                                          device=device, NORM=args.dimensions.norm,subset_key=weights_key )
        
        # Forward pass
        predict = model_cls(x)
        
        results=torch.argmax(predict,dim=1)
        train_acc=accuracy_score(results.cpu(), target.cpu())
        
        # Compute loss
        cls_loss = criterion(predict, target)
        cls_losses.update(cls_loss.item())
        
        # Compute regularization loss
        reg_loss = sum([torch.norm(w, p=2)
                                for w in reconstructed_weights_w])
        reg_loss += sum([torch.norm(w, p=2)
                                for w in reconstructed_weights_b])
        reg_losses.update(reg_loss.item())
        
        # Compute MSE loss
        if f"{hidden_dim}" in gt_model_dict:
            gt_model = gt_model_dict[f"{hidden_dim}"]
            gt_selected_weights_w = [
                w for k, w in gt_model.learnable_parameter.items() if k in weights_key]
            
            gt_selected_weights_b = [
                w for k, w in gt_model.learnable_parameter.items() if k in bias_key]
            
            reconstruct_loss_w = torch.stack([F.mse_loss(
                w, w_gt) for w, w_gt in zip(reconstructed_weights_w, gt_selected_weights_w)])
            
            reconstruct_loss_b = torch.stack([F.mse_loss(
                w, w_gt) for w, w_gt in zip(reconstructed_weights_b, gt_selected_weights_b)])
            reconstruct_loss = torch.mean(torch.stack([reconstruct_loss_w, reconstruct_loss_b]))
        else:
            reconstruct_loss = torch.tensor(0.0)
        reconstruct_losses.update(reconstruct_loss.item())
        
        ce_weight = args.hyper_model.loss_weight.ce_weight
        reg_weight =  args.hyper_model.loss_weight.reg_weight
        recon_weight = args.hyper_model.loss_weight.recon_weight
        
        loss = ce_weight * cls_loss + recon_weight * reconstruct_loss #+ reg_weight * reg_loss
        losses.update(loss.item())
        
        # Zero model_cls grads
        for updated_weight in model_cls.parameters():
            updated_weight.grad = None
        
        # Scale loss and do backward pass
        scaled_loss = loss / (args.experiment.arch_accumulation_steps * args.experiment.batch_accumulation_steps)
        scaled_loss.backward(retain_graph=True)
        torch.autograd.backward(reconstructed_weights_w, [
                        w.grad for k, w in model_cls.named_parameters() if k in weights_key])
        torch.autograd.backward(reconstructed_weights_b, [
                        w.grad for k, w in model_cls.named_parameters() if k in bias_key])
                
        if batch_idx % args.experiment.log_interval == 0 and not args.experiment.debug:
            wandb.log({
                "Running training accuracy argmax" : train_acc,
                "Running average training loss": losses.avg,
                "Cls Loss": cls_losses.avg,
                "Reg Loss": reg_losses.avg,
                "Reconstruct Loss": reconstruct_losses.avg,
                "Learning rate": w_optimizer.param_groups[0]['lr']
                })#, step=batch_idx + (epoch_idx - 1) * len(train_loader) + (block_idx - 1) * max_epochs * len(train_loader))
        
        if batch_idx % args.experiment.log_interval == 0:
            print(f"Iteration {batch_idx}: Loss = {losses.avg:.4f}, Reg Loss = {reg_losses.avg:.4f}, Reconstruct Loss = {reconstruct_losses.avg:.4f}, Cls Loss = {cls_losses.avg:.4f}, Learning rate = {w_optimizer.param_groups[0]['lr']:.4e}")
            
            losses.reset()
            cls_losses.reset()
            reg_losses.reset()
            reconstruct_losses.reset()
            
        if (batch_idx % args.experiment.batch_accumulation_steps) == 0 :
            if args.training.get('clip_grad', 0.0) > 0:
                torch.nn.utils.clip_grad_value_(
                    weights_hypernet.parameters(), args.training.clip_grad)
                
                torch.nn.utils.clip_grad_value_(
                    biases_hypernet.parameters(), args.training.clip_grad)                 
                
            w_optimizer.step()
            w_optimizer.zero_grad()
            b_optimizer.step()        
            b_optimizer.zero_grad()
            step = 0
            #extracted_dim = []
            
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
            
    os.makedirs(args.training.save_model_path, exist_ok=True)
        
    #changed
    hyper_model = get_hypernet(args, 4, device=device)
    
    if args.hyper_model.get('use_ema', True):
        ema = EMA(hyper_model, decay=args.hyper_model.ema_decay)
    else:
        ema = None
        
    criterion, val_criterion, optimizer, scheduler = get_optimizer(args, hyper_model) 

    start_block = 1
    start_epoch = 0
    best_acc = 0.0
    end_epoch = args.experiment.num_epochs + 1 if not args.model.single_block else args.experiment.num_epochs // 4 + 1
    
    frozen_NeRF = get_hypernet(args, 0, device=device)
    
    # If specified, load the checkpoint
    if args.resume_from and "fineTuning" not in args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        
        hyper_model = get_hypernet(args, 4 * (load_trained_blocks(args.resume_from)), device=device)
        
        if args.model.single_block:
            hyper_model = get_hypernet(args, 4, device=device)
            
        criterion, val_criterion, optimizer, scheduler = get_optimizer(args, hyper_model) 
        
        checkpoint_info, hyper_model, optimizer, scheduler, ema = load_checkpoint(args.resume_from, hyper_model, optimizer, scheduler, ema,args=args)
        
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
            #changed
            hyper_model = get_hypernet(args, 4, device=device)
            if(args.model.bottom_up):
                if args.experiment.custom_init:
                    hyper_model = copyParams(frozen_NeRF.model[-4:],hyper_model)
                
                if not args.model.single_block:
                    #changed 
                    tmp_model = get_hypernet(args, 4 * block_id, device=device)   
                    hyper_model=extend_nerf_compose(frozen_NeRF,hyper_model,tmp_model)
                    
                    if args.hyper_model.get('use_ema', True) and args.hyper_model.get('extend_ema', False):
                        ema.extend(hyper_model)
                    
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

            if (args.hyper_model.get('use_ema', True) and not args.hyper_model.get('extend_ema', False)) or (args.hyper_model.get('use_ema', True) and args.model.single_block):
                ema = EMA(hyper_model, decay=args.hyper_model.ema_decay)
            elif not args.hyper_model.get('use_ema', True):
                ema = None
            
            start_epoch = 0
            best_acc = 0.0

            
        dim_dict, gt_model_dict = init_model_dict(args, block_id, args.model.single_block)
        dim_dict = shuffle_coordiates_all(dim_dict)
        
        model_cls, coords_tensor, keys_list, indices_list, size_list, key_mask = dim_dict[f"64"]
        selected_keys = np.unique(keys_list)
        weights_hypernet =  get_hypernet(args, sum('weight' in k for k in selected_keys),args.hyper_model.output_dim, "cuda")
        biases_hypernet =  get_hypernet(args, sum('bias' in k for k in selected_keys),1 , "cuda")
        
        criterion, val_criterion, w_optimizer, w_scheduler = get_optimizer(args, weights_hypernet)
        criterion, val_criterion, b_optimizer, b_scheduler = get_optimizer(args, biases_hypernet)
        
        epoch=None
        
        for epoch in range(start_epoch + 1, end_epoch):
            train_loss = train_one_epoch(weights_hypernet, biases_hypernet, train_loader, w_optimizer, b_optimizer, criterion, dim_dict, gt_model_dict, epoch_idx=epoch, ema=ema, args=args, block_idx=block_id, max_epochs=end_epoch)
            w_scheduler.step()
            b_scheduler.step()

            print(f"Block[{block_id}/{args.model.num_param}]-Epoch[{epoch}/{end_epoch-1}], Training Loss: {train_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

            if epoch % args.experiment.eval_interval == 0 or epoch == 1:
                if ema:
                    ema.apply()

                sampled_model = sample_merge_model(weights_hypernet, biases_hypernet, gt_model_dict[f"{args.dimensions.start}"], args, device=device)
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
                else:
                    save_checkpoint(f"{args.training.save_model_path}/cifar100_nerf_last.pth",hyper_model,optimizer,scheduler,ema,epoch,best_acc, block_id)
        
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
        hyper_model = get_hypernet(args, 0, device=device)
        
        last_hyper_model = get_hypernet(args, 4, device=device)

        for block_id in range(1, args.model.num_param + 1):
            checkpoint_info, last_hyper_model, optimizer, scheduler, ema = load_checkpoint(f"{args.training.save_model_path}/block{block_id}/cifar100_nerf_last.pth", last_hyper_model, optimizer, scheduler ,ema, device=device)
            hyper_model=extend_nerf_compose(hyper_model,last_hyper_model)
        
        if args.hyper_model.get('use_ema', True):
            ema = EMA(hyper_model, decay=args.hyper_model.ema_decay)
        else:
            ema = None
            
        criterion, val_criterion, optimizer, scheduler = get_optimizer(args, hyper_model)
        start_epoch = 0
        best_acc = 0.0
        
        if args.resume_from and "fineTuning" in args.resume_from:
            print(f"Resuming from checkpoint: {args.resume_from}")
            
            hyper_model = get_hypernet(args, 4 * (load_trained_blocks(args.resume_from)), device=device)
            criterion, val_criterion, optimizer, scheduler = get_optimizer(args, hyper_model) 
            checkpoint_info, hyper_model, optimizer, scheduler, ema = load_checkpoint(args.resume_from, hyper_model, optimizer, scheduler, ema,args=args)
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
                
                sampled_model = sample_merge_model(weights_hypernet, biases_hypernet, gt_model_dict[f"{args.dimensions.start}"], args, device=device)
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
    
    if ema:
        ema.apply()
        
    sampled_model = sample_merge_model(weights_hypernet, biases_hypernet, gt_model_dict[f"{args.dimensions.start}"], args, device=device)
    val_loss, val_acc = validate_single(sampled_model, val_loader, val_criterion, args=args, device=device)
    
    if ema:
        ema.restore()
        
    save_checkpoint(f"{args.training.save_model_path}/cifar100_nerf_last.pth",hyper_model,optimizer,scheduler,ema,args.experiment.num_epochs,val_acc, args.model.num_param)
    print("------------------------------------------------------------------------------------------------------------------------------")
    print(f"Checkpoint saved at the end of training with accuracy: {val_acc*100:.2f}%")
    print("------------------------------------------------------------------------------------------------------------------------------")
                    
    if not args.experiment.debug:
        wandb.finish()
    
    print("Training finished.")
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    return
    #testing the best model
    best_hyper_model = get_hypernet(args, number_param, device=device)
    last_hyper_model = get_hypernet(args, number_param, device=device)
    if not args.model.single_block:
        checkpoint_info, best_hyper_model, _, _, best_ema = load_checkpoint(f"{args.training.save_model_path}/block{args.model.num_param}/cifar100_nerf_best.pth", best_hyper_model, optimizer,scheduler ,ema, device=device)
        checkpoint_info, last_hyper_model, _, _, last_ema = load_checkpoint(f"{args.training.save_model_path}/cifar100_nerf_last.pth", last_hyper_model, optimizer,scheduler ,ema, device=device)
    else:
        checkpoint_info, best_hyper_model, _, _, best_ema = load_checkpoint(f"{args.training.save_model_path}/fineTuning/cifar100_nerf_best.pth", best_hyper_model, optimizer,scheduler ,ema, device=device)
        checkpoint_info, last_hyper_model, _, _, last_ema = load_checkpoint(f"{args.training.save_model_path}/cifar100_nerf_last.pth", last_hyper_model, optimizer,scheduler ,ema, device=device)
    
    best_accuracies = []
    last_accuracies = []
    best_accuracies_ema = []
    last_accuracies_ema = []
        
    for hidden_dim in range(args.dimensions.test_range[0], args.dimensions.test_range[1] + 1):
        print(f"--------------------------------------------------------HIDDEN DIM {hidden_dim}--------------------------------------------------------")
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
        best_val_loss, best_val_acc = validate_single(accumulated_model_best, val_loader, val_criterion, args=args, device=device)
        best_accuracies.append(best_val_acc)
        print(f"\tValidation Loss best NeRF: {best_val_loss:.4f}, Validation Accuracy: {best_val_acc*100:.2f}%")
        
        if best_ema:
            best_ema.apply()
            accumulated_model_best_ema = sample_merge_model(best_hyper_model, model, args, K=100, device=device)
            best_val_loss_ema, best_val_acc_ema = validate_single(accumulated_model_best_ema, val_loader, val_criterion, args=args, device=device)
            best_accuracies_ema.append(best_val_acc_ema)
            best_ema.restore()
            print(f"\tValidation Loss best NeRF with EMA: {best_val_loss_ema:.4f}, Validation Accuracy with EMA: {best_val_acc_ema*100:.2f}%")
        
        accumulated_model_last = sample_merge_model(last_hyper_model, model, args, K=100, device=device)
        last_val_loss, last_val_acc = validate_single(accumulated_model_last, val_loader, val_criterion, args=args, device=device)
        last_accuracies.append(last_val_acc)
        print(f"\tValidation Loss last NeRF: {last_val_loss:.4f}, Validation Accuracy: {last_val_acc*100:.2f}%")
        
        if last_ema:
            last_ema.apply()
            accumulated_model_last_ema = sample_merge_model(last_hyper_model, model, args, K=100, device=device)
            last_val_loss_ema, last_val_acc_ema = validate_single(accumulated_model_last_ema, val_loader, val_criterion, args=args, device=device)
            last_accuracies_ema.append(last_val_acc_ema)
            last_ema.restore()
            print(f"\tValidation Loss last NeRF with EMA: {last_val_loss_ema:.4f}, Validation Accuracy with EMA: {last_val_acc_ema*100:.2f}%")

        
    
    best_mean_accuracy = np.mean(best_accuracies)
    best_std_accuracy = np.std(best_accuracies)
    last_mean_accuracy = np.mean(last_accuracies)
    lasy_std_accuracy = np.std(last_accuracies)
    best_mean_accuracy_ema= np.mean(best_accuracies_ema)
    best_std_accuracy_ema = np.std(best_accuracies_ema)
    last_mean_accuracy_ema= np.mean(last_accuracies_ema)
    lasy_std_accuracy_ema = np.std(last_accuracies_ema)
    
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(f"Best NeRF Mean Validation Accuracy: {best_mean_accuracy * 100:.2f}% ± {best_std_accuracy * 100:.2f}%")
    print(f"Last NeRF Mean Validation Accuracy: {last_mean_accuracy * 100:.2f}% ± {lasy_std_accuracy * 100:.2f}%")
    if best_ema:
        print(f"Best NeRF Mean Validation Accuracy with EMA: {best_mean_accuracy_ema * 100:.2f}% ± {best_std_accuracy_ema * 100:.2f}%")
    if last_ema:
        print(f"Last NeRF Mean Validation Accuracy with EMA: {last_mean_accuracy_ema * 100:.2f}% ± {lasy_std_accuracy_ema * 100:.2f}%")
    

    
if __name__ == "__main__":
    args = parse_args()
    print_omegaconf(args)
    
    main_iterative_nerf(args)