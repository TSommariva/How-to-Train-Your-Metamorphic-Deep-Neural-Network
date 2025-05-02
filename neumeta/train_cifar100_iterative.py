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
from neumeta.utils import (AverageMeter, EMA, create_key_masks, get_cifar100, get_cifar10,
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

def verify_weights(model1, model2):
    with torch.no_grad():
        for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
            if name1 != name2:
                print(f"Parameter name mismatch: {name1} vs {name2}")
            else:
                diff = torch.abs(param1 - param2).max().item()
                if diff > 0:
                    print(f"Parameter {name1} differs by {diff}")


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def get_num_workers():
    return 32

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
    for dim in [16]:
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

    return dim_dict, gt_model_dict

def train_one_epoch(model, train_loader, optimizer, criterion, dim_dict, gt_model_dict, epoch_idx ,ema=None, args=None, block_idx=1, max_epochs=200, backbone_parameters={}, layers=3):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    no_accumulation = (args.experiment.arch_accumulation_steps * args.experiment.batch_accumulation_steps) == 1
    step = 0
    
    ce_weight = args.hyper_model.loss_weight.ce_weight
    reg_weight =  args.hyper_model.loss_weight.reg_weight
    recon_weight = args.hyper_model.loss_weight.recon_weight
    
    losses = AverageMeter()
    cls_losses = AverageMeter()
    reg_losses = AverageMeter()
    reconstruct_losses = AverageMeter()
    accuracies = AverageMeter()
    block_flags = [True] * 8#block_idx if block_idx <= args.experiment.simul_blocks else [False] * block_idx
    
    for batch_idx, (x, target) in enumerate(train_loader):
        if device=="cuda" and torch.backends.cudnn.version() >= 7603:
            x, target = x.to(device, memory_format=torch.channels_last), target.to(device)
        else:
            x, target = x.to(device), target.to(device)
        
        step +=1
        if (step != 1) or no_accumulation:
            hidden_dim = random.choice(range(args.dimensions.range[0], args.dimensions.range[1] + 1))
        else:
            hidden_dim = 64
            if block_idx > args.experiment.simul_blocks:
                extracted_blocks = []
                for i in range(args.experiment.simul_blocks):
                    block = random.choice(range(0, block_idx))
                    while block in extracted_blocks:
                        block = random.choice(range(0, block_idx))
                    extracted_blocks.append(block)
                    block_flags[block] = True
        #    extracted_dim.append(hidden_dim)
                        
        model_cls, cls_optimizer ,coords_tensor, keys_list, indices_list, size_list, key_mask = dim_dict[f"{hidden_dim}"]
        selected_keys = np.unique(keys_list)
        cls_optimizer.zero_grad()
        #coords_tensor, keys_list, indices_list, size_list, selected_keys = sample_subset(coords_tensor, keys_list, indices_list, size_list, key_mask, ratio=args.ratio)
        
        #add coordinate noise
        #coords_tensor = coords_tensor + ((torch.rand_like(coords_tensor) - 0.5) * args.training.coordinate_noise).clamp(-0.49, 0.49)
        
        #all the model_cls should share the same backbone_parameters
        with torch.no_grad():
            for name, param in model_cls.named_parameters():
                #if 'alpha' in name or 'fc' in name:
                if 'alpha' in name or 'downsample' in name or (('layer3.8' in name or 'fc' in name) and layers==3):
                    if name in backbone_parameters:
                        model_cls.state_dict()[name].copy_(backbone_parameters[name])
                    else:
                        backbone_parameters[name] = param.detach().clone()
        
        model_cls, reconstructed_weights = sample_weights(model, model_cls,
                                                          coords_tensor, keys_list, indices_list, size_list, key_mask, selected_keys,
                                                          device=device, NORM=args.dimensions.norm, block_flags=block_flags)
        
        model_cls.train()
        
        # Forward pass
        predict = model_cls(x)
        with torch.no_grad():
            results=torch.argmax(predict,dim=1)

            correct = (results == target).sum().item()
            total = target.size(0)

            train_acc=correct / total if total > 0 else 0
            accuracies.update(train_acc)

        # Compute loss
        cls_loss = criterion(predict, target)
        cls_losses.update(cls_loss.item())
        
        # Compute regularization loss
        reg_loss = sum([torch.norm(w, p=2) for w in list(reconstructed_weights.values())])
        reg_losses.update(reg_loss.item())
        
        # Compute MSE loss
        if f"{hidden_dim}" in gt_model_dict:
            gt_selected_weights = [
                w for k, w in gt_model_dict[f"{hidden_dim}"].learnable_parameter.items() if k in selected_keys]
            reconstruct_loss = torch.mean(torch.stack([F.mse_loss(
                w, w_gt) for w, w_gt in zip(list(reconstructed_weights.values()), gt_selected_weights)]))
        else:
            reconstruct_loss = torch.tensor(0.0)
        reconstruct_losses.update(reconstruct_loss.item())
        
        
        loss = ce_weight * cls_loss + reg_weight * reg_loss + recon_weight * reconstruct_loss
        losses.update(loss.item())
        
        # Zero model_cls grads
        for updated_weight in model_cls.parameters():
            updated_weight.grad = None
        
        updated_keys = [k for k in selected_keys if block_flags[int(k.split('.')[1]) - 1]]
        updated_weights = [w for k, w in reconstructed_weights.items() if k in updated_keys]
        
        # Scale loss and do backward pass
        scaled_loss = loss / (args.experiment.arch_accumulation_steps * args.experiment.batch_accumulation_steps)
        scaled_loss.backward(retain_graph=True)
        torch.autograd.backward(updated_weights, [w.grad for k, w in model_cls.named_parameters() if k in updated_keys])
        
                
        cls_optimizer.step()
        model_cls.eval()
        
        with torch.no_grad():
            backbone_parameters = {
                name: model_cls.state_dict()[name].detach().clone()
                for name, _ in model_cls.named_parameters() 
                if 'alpha' in name or 'downsample' in name or (('layer3.8' in name or 'fc' in name) and layers==3)
                #if 'alpha' in name or 'fc' in name
            }
                
        if batch_idx % args.experiment.log_interval == 0 and not args.experiment.debug:
            for i, param_group in enumerate(cls_optimizer.param_groups):
                wandb.log({f"Backbone Learning rate{i}": param_group['lr']}, step=((layers-args.model.first_trained_layer) * args.model.num_param * max_epochs * len(train_loader) // args.experiment.log_interval) + (batch_idx // args.experiment.log_interval) + (epoch_idx - 1) * len(train_loader) // args.experiment.log_interval + (block_idx - args.model.start_block) * max_epochs * len(train_loader) // args.experiment.log_interval)
            
            wandb.log({
                "Running training accuracy argmax" : accuracies.avg,
                "Running average training loss": losses.avg,
                "Cls Loss": cls_losses.avg,
                "Reg Loss": reg_losses.avg,
                "Reconstruct Loss": reconstruct_losses.avg,
                "Learning rate": optimizer.param_groups[0]['lr'],
                }, step=((layers-args.model.first_trained_layer) * args.model.num_param * max_epochs * len(train_loader) // args.experiment.log_interval) + batch_idx // args.experiment.log_interval + (epoch_idx - 1) * len(train_loader) // args.experiment.log_interval + (block_idx - args.model.start_block) * max_epochs * len(train_loader)// args.experiment.log_interval)

                
        if batch_idx % args.experiment.log_interval == 0:
            print(f"Iteration {batch_idx}: Loss = {losses.avg:.4f}, Reg Loss = {reg_losses.avg:.4f}, Reconstruct Loss = {reconstruct_losses.avg:.4f}, Cls Loss = {cls_losses.avg:.4f}, Accuracy = {accuracies.avg*100:.2f}, Learning rate = {optimizer.param_groups[0]['lr']:.4e}")

        if (batch_idx % args.experiment.batch_accumulation_steps) == 0 :
            if args.training.get('clip_grad', 0.0) > 0:
                torch.nn.utils.clip_grad_value_(
                    model.parameters(), args.training.clip_grad)                
                
            optimizer.step()        
            optimizer.zero_grad(set_to_none=True)
            step = 0
            
            #if ema:
            #    ema.update()  # Update the EMA after each training step
    
    #if step > 0:
    #    optimizer.step()        
    #    if ema:
    #        ema.update()    
    #tr_loss, tr_acc = validate_single(model_cls, train_loader, nn.CrossEntropyLoss(), args=args, device=device)
    if not args.experiment.debug:
        wandb.log({
                    "trainLoss_last model of the epoch" : losses.avg,
                    "trainAcc_last model of the epoch" : accuracies.avg
                }, step=((layers-args.model.first_trained_layer) * args.model.num_param * max_epochs * len(train_loader) // args.experiment.log_interval) + batch_idx // args.experiment.log_interval + (epoch_idx - 1) * len(train_loader) // args.experiment.log_interval + (block_idx - args.model.start_block) * max_epochs * len(train_loader) // args.experiment.log_interval )
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
                         smooth=args.model.smooth, fuse=args.model.smooth,
                         config_args=args,first_meta_layer=args.model.start_layer,num_layers=3).to(device)
    
    print("Maximum DIM: ",find_max_dim(model))

    val_loss, acc = validate_single(model, val_loader, nn.CrossEntropyLoss(), args=args, device=device)
    print(f"Initial Permutated model Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc*100:.2f}%")
    
    checkpoint = model.learnable_parameter
    number_param = len(checkpoint)
    print(f"Number of parameters to be learned: {number_param}")
    print(f"Parameters keys: {model.keys}")
            
    os.makedirs(args.training.save_model_path, exist_ok=True)
    
    ema=None
    start_block = args.model.start_block
    start_epoch = 0
    best_acc = 0.0
    end_epoch = args.experiment.num_epochs + 1
    backbone_parameters = {}
    prev_NeRF = get_hypernet(args, 0,total_param=number_param ,device=device)
    trained_layers = args.model.first_trained_layer
    #dictionaries = []
    #dic = torch.load("/homes/tsommariva/tmp/model_dictionary.pth", weights_only=False)
    #for i in range(21):
    #    tmp_dic = copy.deepcopy(dic)
    #    dictionaries.append(tmp_dic)
        
    
    # If specified, load the checkpoint
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        if args.resume_from == "/work/tesi_tsommariva/experiments/Paper/FULL_allLayersSim:True_Blocks1to8_from2557846/layer3/nerf_block4_best.pth":
            resume_from="/work/tesi_tsommariva/experiments/Paper/FULL_allLayersSim:True_Blocks1to8_from2557846/layer3/nerf_block4_best.pth"
        else:
            resume_from=args.resume_from
        trained_layers, trained_blocks = load_trained_blocks(resume_from)
        dim_dict, gt_model_dict = init_model_dict(args, trained_blocks, args.model.single_block, first_meta_layer=args.model.start_layer, num_layers=trained_layers)
        dim_dict = shuffle_coordiates_all(dim_dict)
        _, _, _, keys_list, _, _, _ = dim_dict[f"{64}"]
        selected_keys = np.unique(keys_list)
        hyper_model = get_hypernet(args, 4 * trained_blocks + (4 * (trained_layers - args.model.start_layer) * args.model.num_param),total_param=number_param ,key_list=selected_keys ,device=device)
        
        if args.hyper_model.get('use_ema', True):
            ema = EMA(hyper_model, decay=args.hyper_model.ema_decay)
        else:
            ema = None
    
        criterion, val_criterion, optimizer, scheduler = get_optimizer(args, hyper_model, first_block=trained_blocks==1) 
        
        checkpoint_info, hyper_model, optimizer, scheduler, ema = load_checkpoint(resume_from, hyper_model, optimizer, scheduler, None, args=args)
        
        if optimizer is None:
            criterion, val_criterion, optimizer, scheduler = get_optimizer(args, hyper_model, first_block=trained_blocks==1)
        
        start_epoch = checkpoint_info['epoch']
        best_acc = checkpoint_info['best_acc']
        start_block = trained_blocks
        backbone_parameters = checkpoint_info['backbone_parameters']
        if not args.experiment.all_layers_sim:
            dim_dict, gt_model_dict = init_model_dict(args, trained_blocks, args.model.single_block, first_meta_layer=trained_layers, num_layers=trained_layers)
            dim_dict = shuffle_coordiates_all(dim_dict)
        sampled_model = sample_merge_model(hyper_model, dim_dict[f"{args.dimensions.start}"][0], args, backbone_parameters=backbone_parameters ,device=device)
        train_loss, train_acc = validate_single(sampled_model, train_loader, val_criterion, args=args, device=device)
        val_loss, val_acc = validate_single(sampled_model, val_loader, val_criterion, args=args, device=device)
        print(f"Loaded Model, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc*100:.2f}%")
        a_diff = abs(val_acc - best_acc)*100
        print(f"Validation accuracy difference: {a_diff:.2f}%")
        print("------------------------------------------------------------------------------------------------------------------------------")
        
        if not args.experiment.debug:    
            wandb.log({
                "Train Loss_model sampled outside training": train_loss,
                "Train Accuracy_model sampled outside training": train_acc,
                "Validation Loss_model sampled outside training": val_loss,
                "Validation Accuracy_model sampled outside training": val_acc
            }, step=((trained_layers-args.model.first_trained_layer) * args.model.num_param * args.experiment.num_epochs * len(train_loader) // args.experiment.log_interval) + (start_epoch) * len(train_loader) // args.experiment.log_interval + (start_block - args.model.start_block) * args.experiment.num_epochs * len(train_loader) // args.experiment.log_interval)
                
        if a_diff > 0.5:
            return -2
        del checkpoint_info
        del sampled_model
        gc.collect()

        print(f"Resuming from layer: {trained_layers} block: {start_block}, epoch: {start_epoch}, best accuracy: {best_acc*100:.2f}%")
        # Note: If there are more elements to retrieve, do so here.  

    for layers in range(trained_layers, 4):
        print(f"LAYER[{layers}/3]")
        os.makedirs(f"{args.training.save_model_path}/layer{layers}", exist_ok=True)
        num_param = args.model.num_param
        if layers==3 and num_param>=7:
            num_param=7
        start_layer = args.model.start_layer if args.experiment.all_layers_sim else layers
        for block_id in range(start_block, num_param + 1):
            print(f"BLOCK[{block_id}/{args.model.num_param}]")

            if not (args.resume_from and block_id == start_block and layers == trained_layers):
                dim_dict, gt_model_dict = init_model_dict(args, block_id, args.model.single_block,first_meta_layer=args.model.start_layer,num_layers=layers)
                dim_dict = shuffle_coordiates_all(dim_dict)
                _, _, _, keys_list, _, _, _ = dim_dict[f"{64}"]
                selected_keys = np.unique(keys_list)
                if block_id == start_block and layers==1:
                    hyper_model = get_hypernet(args, 4 * block_id + (4 * (layers - args.model.start_layer) * args.model.num_param),total_param=number_param ,key_list=selected_keys ,device=device)
                else:
                    if block_id == start_block:
                        hyper_model = extend_nerf_compose(prev_NeRF,False, args, 4 * block_id + (4 * (layers - args.model.start_layer) * args.model.num_param),total_param=number_param,key_list=selected_keys ,device=device)
                    else:
                        hyper_model = extend_nerf_compose(prev_NeRF,args.experiment.custom_init, args, 4 * block_id + (4 * (layers - args.model.start_layer) * args.model.num_param), total_param=number_param,key_list=selected_keys ,device=device)
                    start_epoch = 0
                    best_acc = 0.0 

                criterion, val_criterion, optimizer, scheduler = get_optimizer(args, hyper_model, first_block=(block_id==start_block))   
                if not args.experiment.all_layers_sim and layers != args.model.start_layer:
                    dim_dict, gt_model_dict = init_model_dict(args, block_id, args.model.single_block,first_meta_layer=start_layer,num_layers=layers)
                    dim_dict = shuffle_coordiates_all(dim_dict)
            
                _, _, _, keys_list, _, _, _ = dim_dict[f"{64}"]
                selected_keys = np.unique(keys_list)
                print(f"key learned in this iteration: {selected_keys}")     
                gc.collect()
                
            epoch=None
            end_epoch = args.experiment.num_epochs + 1
            for epoch in range(start_epoch + 1, end_epoch):
                train_loss, train_acc, backbone_parameters = train_one_epoch(hyper_model, train_loader, optimizer, criterion, dim_dict, gt_model_dict, epoch_idx=epoch, ema=ema, args=args, block_idx=block_id, max_epochs=args.experiment.num_epochs, backbone_parameters=backbone_parameters,layers=layers)
                scheduler.step()
                hyper_model.eval()

                print(f"LAYER[{layers}/3]-Block[{block_id}/{args.model.num_param}]-Epoch[{epoch}/{end_epoch-1}], Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc*100:.2f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

                if epoch % args.experiment.eval_interval == 0 or epoch == 1:
                    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    if (epoch == end_epoch - 1) or (epoch % 50 == 0):
                        test_dims = [16,32,48,64]
                    else:
                        test_dims = [64]
                    for dim in test_dims:
                        sampled_model = sample_merge_model(hyper_model, dim_dict[f"{dim}"][0], args, backbone_parameters=backbone_parameters ,device=device, K=100)
                        val_loss, val_acc = validate_single(sampled_model, val_loader, val_criterion, args=args, device=device)
                        train_loss, train_acc = validate_single(sampled_model, train_loader, val_criterion, args=args, device=device)

                        if not args.experiment.debug and dim == 64:    
                            wandb.log({
                                "Train Loss_model sampled outside training": train_loss,
                                "Train Accuracy_model sampled outside training": train_acc,
                                "Validation Loss_model sampled outside training": val_loss,
                                "Validation Accuracy_model sampled outside training": val_acc
                            }, step=((layers-args.model.first_trained_layer) * args.model.num_param * args.experiment.num_epochs * len(train_loader) // args.experiment.log_interval) + (epoch) * len(train_loader) // args.experiment.log_interval + (block_id - args.model.start_block) * args.experiment.num_epochs * len(train_loader) // args.experiment.log_interval)
                        elif not args.experiment.debug:
                            wandb.log({
                                f"Dim{dim} - Train Loss_model sampled outside training": train_loss,
                                f"Dim{dim} - Train Accuracy_model sampled outside training": train_acc,
                                f"Dim{dim} - Validation Loss_model sampled outside training": val_loss,
                                f"Dim{dim} - Validation Accuracy_model sampled outside training": val_acc
                            }, step=((layers-args.model.first_trained_layer) * args.model.num_param * args.experiment.num_epochs * len(train_loader) // args.experiment.log_interval) + (epoch) * len(train_loader) // args.experiment.log_interval + (block_id - args.model.start_block) * args.experiment.num_epochs * len(train_loader) // args.experiment.log_interval)
                        print(f"Hidden Dim = {dim} - LAYER[{layers}/3]-Block[{block_id}/{args.model.num_param}]-Epoch[{epoch}/{end_epoch-1}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc*100:.2f}%")
                        print(f"Hidden Dim = {dim} - LAYER[{layers}/3]-Block[{block_id}/{args.model.num_param}]-Epoch[{epoch}/{end_epoch-1}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc*100:.2f}%")
                        print("------------------------------------------------------------------------------------------------------------------------------")

                        # Save the checkpoint
                        if val_acc >= best_acc and dim == 64:
                            best_acc = val_acc
                            save_checkpoint(f"{args.training.save_model_path}/layer{layers}/nerf_block{block_id}_best.pth",hyper_model,optimizer,scheduler,ema,epoch,val_acc, trained_blocks=block_id,trained_layers=layers ,backbone_parameters=backbone_parameters)
                            print(f"LAYER[{layers}/3]-Block[{block_id}/{args.model.num_param}] Checkpoint saved at epoch {epoch} with accuracy: {val_acc*100:.2f}%; best accuracy: {best_acc*100:.2f}%")
                        #elif val_acc < 0.1:
                        #    return -1
                        elif dim==64:
                            print(f"LAYER[{layers}/3]-Block[{block_id}/{args.model.num_param}], best accuracy: {best_acc*100:.2f}%")
                    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

                    #torch.cuda.empty_cache()

                #if train_acc < 0.7:
                #    return -4

            if end_epoch == start_epoch + 1:
                #torch.cuda.empty_cache()        
                prev_NeRF = hyper_model
                continue
            
            prev_NeRF = hyper_model
        
        if args.resume_from:
            start_block = args.model.start_block          
            
        
                    
        
    print("Training finished.")
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    #print(hyper_model)
    #summary(hyper_model, (6,))
    return 0

def test(args):
    num_workers = get_num_workers()
    _, val_loader = get_cifar100(args.training.batch_size, num_workers)

    model = create_model(args.model.type, 
                         hidden_dim=args.dimensions.start,
                         num_param=args.model.num_param,
                         bottom_up=args.model.bottom_up,
                         path=args.model.pretrained_path, 
                         smooth=args.model.smooth, fuse=args.model.smooth,
                         config_args=args, first_meta_layer=args.model.start_layer, num_layers=3).to(device)

    
    checkpoint = model.learnable_parameter
    number_param = len(checkpoint)
    
    best_hyper_model = get_hypernet(args, number_param,total_param = number_param,key_list = model.keys, device=device)
    
    criterion, _, _, _ = get_optimizer(args, best_hyper_model, first_block=False) 
    if args.experiment.test == True:
        test_path = args.experiment.test_path
    else:
        test_path = args.training.save_model_path
        
    checkpoint_info, best_hyper_model, _, _, _ = load_checkpoint(f"{test_path}/layer3/nerf_block7_best.pth", best_hyper_model, None,None ,None, device=device)
    if checkpoint_info is None:
        checkpoint_info, best_hyper_model, _, _, _ = load_checkpoint(f"{test_path}/layer2/nerf_block8_best.pth", best_hyper_model, None,None ,None, device=device)
    if checkpoint_info is None:
        checkpoint_info, best_hyper_model, _, _, _ = load_checkpoint(f"{test_path}/layer1/nerf_block8_best.pth", best_hyper_model, None,None ,None, device=device)
    if checkpoint_info is None:
        checkpoint_info, best_hyper_model, _, _, _ = load_checkpoint(f"{test_path}/nerf_block7_best.pth", best_hyper_model, None,None ,None, device=device)
    if checkpoint_info is None:
        checkpoint_info, best_hyper_model, _, _, _ = load_checkpoint(f"{test_path}/nerf_block7.pth", best_hyper_model, None,None ,None, device=device)
    if checkpoint_info is None:
        checkpoint_info, best_hyper_model, _, _, _ = load_checkpoint(f"{test_path}/nerf_block8.pth", best_hyper_model, None,None ,None, device=device)
    if checkpoint_info is None:
        checkpoint_info, best_hyper_model, _, _, _ = load_checkpoint(f"{test_path}/nerf_block8_best.pth", best_hyper_model, None,None ,None, device=device)
    if checkpoint_info is None:
        checkpoint_info, best_hyper_model, _, _, _ = load_checkpoint(f"{test_path}/block8/cifar100_nerf_best.pth", best_hyper_model, None,None ,None, device=device)
    if checkpoint_info is None:
        checkpoint_info, best_hyper_model, _, _, _ = load_checkpoint(f"{test_path}/block8/cifar100_nerf_last.pth", best_hyper_model, None,None ,None, device=device)
    if checkpoint_info is None:
        return -5
    backbone_parameters = checkpoint_info['backbone_parameters']
    start_epoch = checkpoint_info['epoch']
    best_acc = checkpoint_info['best_acc']
    start_block = args.model.num_param
    print(f"Testing: Prev Training : block: {start_block}, epoch: {start_epoch}, accuracy: {best_acc*100:.2f}%")
    best_hyper_model.eval()
    #save_checkpoint(f"{test_path}/nerf_block7_v2.pth",best_hyper_model,optimizer,scheduler,None,50,checkpoint_info['epoch'], trained_blocks=7, backbone_parameters=backbone_parameters)
    hyper_model_type = args.hyper_model.get('type', 'mlp')
    if hyper_model_type == 'resmlpDict':
        tmphyp = get_hypernet(args, (args.model.num_param - args.model.start_block + 1 )* 4, total_param=number_param, device=device, hyper_model_type='resmlp')
    else:
        tmphyp = get_hypernet(args, (args.model.num_param - args.model.start_block + 1) * 4, total_param=number_param, device=device)
    summary(tmphyp, (6,))
    del tmphyp
    gc.collect()

    del checkpoint_info
    del model
    del _
    gc.collect()
    torch.cuda.empty_cache()    
    for hidden_dim in [16,32,48,64]:
         model = create_model(args.model.type, 
                                 hidden_dim=hidden_dim,
                                 num_param=args.model.num_param,
                                 bottom_up=args.model.bottom_up,
                                 single_block=False,
                                 path=args.model.pretrained_path, 
                                 smooth=args.model.smooth, fuse=args.model.fuse,
                                 prior=False, config_args=args, first_meta_layer=args.model.start_layer, num_layers=3)
         if device=="cuda" and torch.backends.cudnn.version() >= 7603:
             model = model.to(device, memory_format=torch.channels_last)  # Module parameters need to be channels last
         else:
             model = model.to(device)
         
         model.eval()
         # Sample the merged model for K times
         accumulated_model = sample_merge_model(best_hyper_model, model, args,backbone_parameters=backbone_parameters ,K=100, device=device)
         val_loss, val_acc = validate_single(accumulated_model, val_loader, criterion, args, device=device)
         print(f"Best Model, Dimension:{hidden_dim} Validation loss:{val_loss:.4f} Validation Accuracy:{val_acc*100:.2f}")
         
    print("------------------------------------------------------------------------------------------------------------------------------")
    
    last_checkpoint_info, last_hyper_model, _, _, _ = load_checkpoint(f"{test_path}/nerf_block7_last.pth", best_hyper_model, None, None ,None, device=device)
    if last_checkpoint_info is None:
        last_checkpoint_info, last_hyper_model, _, _, _ = load_checkpoint(f"{test_path}/nerf_block8_last.pth", best_hyper_model, None, None ,None, device=device)
    if last_checkpoint_info is not None:
        last_backbone_parameters = last_checkpoint_info['backbone_parameters']
        for hidden_dim in [16,32,48,64]:
             model = create_model(args.model.type, 
                                     hidden_dim=hidden_dim,
                                     num_param=args.model.num_param,
                                     bottom_up=args.model.bottom_up,
                                     single_block=False,
                                     path=args.model.pretrained_path, 
                                     smooth=args.model.smooth, fuse=args.model.fuse,
                                     prior=False, config_args=args, first_meta_layer=args.model.start_layer, num_layers=3)
             if device=="cuda" and torch.backends.cudnn.version() >= 7603:
                 model = model.to(device, memory_format=torch.channels_last)  # Module parameters need to be channels last
             else:
                 model = model.to(device)

             model.eval()
             # Sample the merged model for K times
             accumulated_model = sample_merge_model(last_hyper_model, model, args,backbone_parameters=last_backbone_parameters ,K=100, device=device)
             val_loss, val_acc = validate_single(accumulated_model, val_loader, criterion, args, device=device)
             print(f"Last Model, Dimension:{hidden_dim} Validation loss:{val_loss:.4f} Validation Accuracy:{val_acc*100:.2f}")

        print("------------------------------------------------------------------------------------------------------------------------------")
    
    start_time = time.time()
    validate_all_dimensions(best_hyper_model, backbone_parameters, args.model.num_param, val_loader, criterion, create_model, args, device='cuda', step = 2, first_meta_layer=args.model.start_layer, num_layers=3)
    elapsed_time = (time.time() - start_time)/60
    print(f"Time elapsed for testing: {elapsed_time:.2f} minutes")

    return 0


if __name__ == "__main__":
    args = parse_args()
    print_omegaconf(args)
    
    if not args.experiment.debug:
        initialize_wandb(args)

#
    if args.experiment.test == False:
        exit_code = main_iterative_nerf(args)
    else:
        exit_code = 0
    
    if exit_code == 0:
        exit_code = test(args)
#        
        
    if not args.experiment.debug:
        wandb.finish()   
         
    sys.exit(exit_code)