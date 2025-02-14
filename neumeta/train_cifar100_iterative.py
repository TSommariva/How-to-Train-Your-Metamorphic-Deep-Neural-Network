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
                model_trained = model_trained.to(device, memory_format=torch.channels_last)  # Module parameters need to be channels last
            else:
                model_trained = model_trained.to(device)
            model_trained.eval()
            
            gt_model_dict[f"{dim}"] = model_trained
    return dim_dict, gt_model_dict

def train_one_epoch(model, train_loader, optimizer, criterion, dim_dict, gt_model_dict, epoch_idx ,ema=None, args=None, block_idx=1, max_epochs=200, backbone_parameters={}):
    model.train()
    optimizer.zero_grad()
    
    no_accumulation = (args.experiment.arch_accumulation_steps * args.experiment.batch_accumulation_steps) == 1
    step = 0
    
    losses = AverageMeter()
    cls_losses = AverageMeter()
    reg_losses = AverageMeter()
    reconstruct_losses = AverageMeter()
    accuracies = AverageMeter()
    
    block_flags = [True] * block_idx if block_idx <= 3 else [False] * block_idx
    
    for batch_idx, (x, target) in enumerate(train_loader):
        if device=="cuda" and torch.backends.cudnn.version() >= 7603:
            x, target = x.to(device, memory_format=torch.channels_last), target.to(device)
        else:
            x, target = x.to(device), target.to(device)
        
        step +=1
        if (step != 1) or no_accumulation:
            hidden_dim = random.choice(range(args.dimensions.range[0], args.dimensions.range[1] + 1))
        else:
            hidden_dim = 256
            if block_idx > 3:
                extracted_blocks = []
                for i in range(3):
                    block = random.choice(range(0, block_idx))
                    while block not in extracted_blocks:
                        block = random.choice(range(0, block_idx))
                    extracted_blocks.append(block)
                    block_flags[block] = True
                    
        #    extracted_dim.append(hidden_dim)
                        
        model_cls, cls_optimizer ,coords_tensor, keys_list, indices_list, size_list, key_mask = dim_dict[f"{hidden_dim}"]
        selected_keys = np.unique(keys_list)
        #coords_tensor, keys_list, indices_list, size_list, selected_keys = sample_subset(coords_tensor, keys_list, indices_list, size_list, key_mask, ratio=args.ratio)
        
        #add coordinate noise
        #coords_tensor = coords_tensor + ((torch.rand_like(coords_tensor) - 0.5) * args.training.coordinate_noise).clamp(-0.49, 0.49)
        
        #all the model_cls should share the same backbone_parameters
        with torch.no_grad():
            for name, param in model_cls.named_parameters():
                #if name not in model_cls.learnable_parameter.keys():
                if 'alpha' in name or 'fc' in name:
                    if name in backbone_parameters:
                        param.copy_(backbone_parameters[name].clone())
                    else:
                        backbone_parameters[name] = param.clone()
        
        model_cls, reconstructed_weights = sample_weights(model, model_cls,
                                                      coords_tensor, keys_list, indices_list, size_list, key_mask, selected_keys,
                                                          device=device, NORM=args.dimensions.norm, block_flags=block_flags)
        
        model_cls.train()
        
        # Forward pass
        predict = model_cls(x)
        results=torch.argmax(predict,dim=1)
        
        correct = (results == target).sum().item()
        total = target.size(0)

        train_acc=correct / total if total > 0 else 0
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
        
                
        cls_optimizer.step()
        
        with torch.no_grad():
            backbone_parameters = {
                name: param.clone() 
                for name, param in model_cls.named_parameters() 
                if 'alpha' in name or 'fc' in name
                #if name not in model_cls.learnable_parameter.keys()
            }
                
        if batch_idx % args.experiment.log_interval == 0 and not args.experiment.debug:
            for i, param_group in enumerate(cls_optimizer.param_groups):
                wandb.log({f"Backbone Learning rate{i}": param_group['lr']}, step=(batch_idx // args.experiment.log_interval) + (epoch_idx - 1) * len(train_loader) // args.experiment.log_interval + (block_idx - 1) * max_epochs * len(train_loader) // args.experiment.log_interval)
            
            wandb.log({
                "Running training accuracy argmax" : accuracies.avg,
                "Running average training loss": losses.avg,
                "Cls Loss": cls_losses.avg,
                "Reg Loss": reg_losses.avg,
                "Reconstruct Loss": reconstruct_losses.avg,
                "Learning rate": optimizer.param_groups[0]['lr'],
                }, step=batch_idx // args.experiment.log_interval + (epoch_idx - 1) * len(train_loader) // args.experiment.log_interval + (block_idx - 1) * max_epochs * len(train_loader)// args.experiment.log_interval)

                
        if batch_idx % args.experiment.log_interval == 0:
            print(f"Iteration {batch_idx}: Loss = {losses.avg:.4f}, Reg Loss = {reg_losses.avg:.4f}, Reconstruct Loss = {reconstruct_losses.avg:.4f}, Cls Loss = {cls_losses.avg:.4f}, Accuracy = {accuracies.avg*100:.2f}, Learning rate = {optimizer.param_groups[0]['lr']:.4e}")

        if (batch_idx % args.experiment.batch_accumulation_steps) == 0 :
            if args.training.get('clip_grad', 0.0) > 0:
                torch.nn.utils.clip_grad_value_(
                    model.parameters(), args.training.clip_grad)                
                
            optimizer.step()        
            optimizer.zero_grad()
            step = 0
            
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
                }, step=batch_idx // args.experiment.log_interval + (epoch_idx - 1) * len(train_loader) // args.experiment.log_interval + (block_idx - 1) * max_epochs * len(train_loader) // args.experiment.log_interval )
    return losses.avg, accuracies.avg, backbone_parameters
    
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

    start_block = 1
    start_epoch = 0
    best_acc = 0.0
    end_epoch = args.experiment.num_epochs + 1 if not args.model.single_block else args.experiment.num_epochs // 4 + 1
    backbone_parameters = {}
    
    prev_NeRF = get_hypernet(args, 0,total_param=number_param ,device=device)
    
    # If specified, load the checkpoint
    if args.resume_from:
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
        start_block = trained_blocks
        backbone_parameters = checkpoint_info['backbone_parameters']
        print(f"Resuming from block: {start_block}, epoch: {start_epoch}, best accuracy: {best_acc*100:.2f}%")
        # Note: If there are more elements to retrieve, do so here.  

    if not args.experiment.debug:
        initialize_wandb(args)
    
    for block_id in range(start_block, args.model.num_param + 1):
        #os.makedirs(f"{args.training.save_model_path}/block{block_id}", exist_ok=True)
        print(f"BLOCK[{block_id}/{args.model.num_param}]")
        
        if not (args.resume_from and block_id == start_block):
            dim_dict, gt_model_dict = init_model_dict(args, block_id, args.model.single_block)
            dim_dict = shuffle_coordiates_all(dim_dict)
            _, _, _, keys_list, _, _, _ = dim_dict[f"{256}"]
            selected_keys = np.unique(keys_list)
            hyper_model = get_hypernet(args, 4 * block_id,total_param=number_param ,key_list=selected_keys ,device=device)

            if block_id != start_block:
                hyper_model=extend_nerf_compose(prev_NeRF,hyper_model,args.experiment.custom_init)
                start_epoch = 0
                best_acc = 0.0 

            criterion, val_criterion, optimizer, scheduler = get_optimizer(args, hyper_model, first_block=block_id==start_block)   

            if args.hyper_model.get('use_ema', True):
                if block_id == start_block or not args.hyper_model.get('extend_ema', False):
                    ema = EMA(hyper_model, decay=args.hyper_model.ema_decay)
                else:
                    ema.extend(hyper_model)
            else:
                ema=None
        
        epoch=None
        
        for epoch in range(start_epoch + 1, end_epoch):
            train_loss, train_acc, backbone_parameters = train_one_epoch(hyper_model, train_loader, optimizer, criterion, dim_dict, gt_model_dict, epoch_idx=epoch, ema=ema, args=args, block_idx=block_id, max_epochs=args.experiment.num_epochs, backbone_parameters=backbone_parameters)
            scheduler.step()

            print(f"Block[{block_id}/{args.model.num_param}]-Epoch[{epoch}/{end_epoch-1}], Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc*100:.2f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

            if epoch % args.experiment.eval_interval == 0 or epoch == 1:
                if ema:
                    ema.apply()

                sampled_model = sample_merge_model(hyper_model, dim_dict[f"{args.dimensions.start}"][0], args, backbone_parameters=backbone_parameters ,device=device)
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
                    }, step=(epoch) * len(train_loader) // args.experiment.log_interval + (block_id - 1) * args.experiment.num_epochs * len(train_loader) // args.experiment.log_interval)
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print(f"Block[{block_id}/{args.model.num_param}]-Epoch[{epoch}/{end_epoch-1}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc*100:.2f}%")
                print(f"Block[{block_id}/{args.model.num_param}]-Epoch[{epoch}/{end_epoch-1}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc*100:.2f}%")
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

                # Save the checkpoint
                if val_acc > best_acc:
                    best_acc = val_acc
                    save_checkpoint(f"{args.training.save_model_path}/cifar100_nerf_best.pth",hyper_model,optimizer,scheduler,ema,epoch,best_acc, trained_blocks=block_id, backbone_parameters=backbone_parameters)
                    print("------------------------------------------------------------------------------------------------------------------------------")
                    print(f"Block[{block_id}/{args.model.num_param}] Checkpoint saved at epoch {epoch} with accuracy: {best_acc*100:.2f}%")
                    print("------------------------------------------------------------------------------------------------------------------------------")
                else:
                    save_checkpoint(f"{args.training.save_model_path}/cifar100_nerf_last.pth",hyper_model,optimizer,scheduler,ema,epoch,best_acc, trained_blocks=block_id, backbone_parameters=backbone_parameters)
        
        if epoch is None:
            epoch = end_epoch
            
        if args.model.single_block:
            save_checkpoint(f"{args.training.save_model_path}/cifar100_nerf_last.pth",hyper_model,optimizer,scheduler,ema,epoch,best_acc, block_id,backbone_parameters=backbone_parameters)
            print("------------------------------------------------------------------------------------------------------------------------------")
            print(f"Block[{block_id}/{args.model.num_param}] Checkpoint saved at the end of block {block_id} with accuracy: {best_acc*100:.2f}%")
            print("------------------------------------------------------------------------------------------------------------------------------")
        
        start_time = time.time()
        validate_all_dimensions(hyper_model, backbone_parameters, block_id, val_loader, criterion, create_model, args, device='cuda')
        elapsed_time = (time.time() - start_time)/60
        print(f"Time elapsed for validate_all_dimensions: {elapsed_time:.2f} minutes")
        prev_NeRF = hyper_model
        
    if ema:
        ema.apply()
        
    sampled_model = sample_merge_model(hyper_model, dim_dict[f"{args.dimensions.start}"][0], args,backbone_parameters=backbone_parameters ,device=device)
    val_loss, val_acc = validate_single(sampled_model, val_loader, val_criterion, args=args, device=device)
    
    if ema:
        ema.restore()
        
    save_checkpoint(f"{args.training.save_model_path}/cifar100_nerf_last.pth",hyper_model,optimizer,scheduler,ema,args.experiment.num_epochs,val_acc, trained_blocks=args.model.num_param, backbone_parameters=backbone_parameters)
    print("------------------------------------------------------------------------------------------------------------------------------")
    print(f"Checkpoint saved at the end of training with accuracy: {val_acc*100:.2f}%")
    print("------------------------------------------------------------------------------------------------------------------------------")
                    
    if not args.experiment.debug:
        wandb.finish()
    
    print("Training finished.")
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    #testing the best model
    best_hyper_model = get_hypernet(args, number_param,total_param = number_param ,key_list = model.keys, device=device)
    checkpoint_info, best_hyper_model, _, _, best_ema = load_checkpoint(f"{args.training.save_model_path}/cifar100_nerf_best.pth", best_hyper_model, optimizer,scheduler ,ema, device=device)
    best_backbone_parameters = checkpoint_info['backbone_parameters']
    best_accuracies = []
    if best_ema:
            best_ema.apply()

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
                                prior=False)
        
        if device=="cuda" and torch.backends.cudnn.version() >= 7603:
            model = model.to(device, memory_format=torch.channels_last)  # Module parameters need to be channels last
        else:
            model = model.to(device)
        
        # Sample the merged model for K times
        accumulated_model_best = sample_merge_model(best_hyper_model, model, args,backbone_parameters=best_backbone_parameters ,K=100, device=device)
        best_val_loss, best_val_acc = validate_single(accumulated_model_best, val_loader, val_criterion, args=args, device=device)
        best_accuracies.append(best_val_acc)
        print(f"\tValidation Loss best NeRF: {best_val_loss:.4f}, Validation Accuracy: {best_val_acc*100:.2f}%")

    best_mean_accuracy = np.mean(best_accuracies)
    best_std_accuracy = np.std(best_accuracies)
    
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(f"Best NeRF Mean Validation Accuracy: {best_mean_accuracy * 100:.2f}% ± {best_std_accuracy * 100:.2f}%")
    

if __name__ == "__main__":
    args = parse_args()
    print_omegaconf(args)
    
    main_iterative_nerf(args)