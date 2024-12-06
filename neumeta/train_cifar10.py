# Import necessary libraries
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
# Import functions from neumeta module
from neumeta.models import create_model_cifar10 as create_model
from neumeta.utils import (AverageMeter, EMA, load_checkpoint, print_omegaconf, 
                       sample_coordinates, sample_merge_model, 
                       sample_subset, sample_weights, save_checkpoint, 
                       set_seed, shuffle_coordiates_all, 
                       # validate, validate_ensemble,validate_merge,
                       validate_single, get_cifar10, sample_single_model,
                       get_hypernet, get_optimizer,
                       parse_args, 
                       weighted_regression_loss)
from neumeta.models import BasicBlock, BasicBlock_Resize
from sklearn.metrics import accuracy_score

# Print message
print("Training INR On CIFAR10")

# Set device to GPU if available, else CPU
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Function to find the maximum dimension of the model
def find_max_dim(model_cls):
    # Get the learnable parameters of the model
    checkpoint = model_cls.learnable_parameter
    # Set the maximum value to the length of the checkpoint
    max_value = len(checkpoint)
    # Iterate over the new model's weights
    for i, (k, tensor) in enumerate(checkpoint.items()):
        # Handle 2D tensors (e.g., weight matrices)
        if len(tensor.shape) == 4:
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
        config (dict): Configuration parameters for the run.
    """
    # Name the run using current time and configuration name
    run_name = f"{time.strftime('%Y%m%d%H%M%S')}-{config.experiment.name}"
    
    wandb.init(project="ninr", name=run_name, config=dict(config), group='cifar10')

# Function to train the model for one epoch
def train_one_epoch(model, train_loader, optimizer, criterion, dim_dict, gt_model_dict, epoch_idx, ema=None, args=None):
    # Set the model to training mode
    model.train()
    total_loss = 0.0

    # Initialize AverageMeter objects to track the losses
    losses = AverageMeter()
    cls_losses = AverageMeter()
    reg_losses = AverageMeter()
    reconstruct_losses = AverageMeter()

    # Iterate over the training data
    for batch_idx, (x, target) in enumerate(train_loader):
        # Zero the gradients
        optimizer.zero_grad()
        # Move the data to the device
        x, target = x.to(device), target.to(device)
        # Choose a random hidden dimension
        hidden_dim = random.choice(args.dimensions.range)
        if batch_idx % 20 == 0:
            hidden_dim = 64
        # Get the model class, coordinates, keys, indices, size, and key mask for the chosen dimension
        model_cls, coords_tensor, keys_list, indices_list, size_list, key_mask = dim_dict[f"{hidden_dim}"]
        # Sample a subset of the coordinates, keys, indices, size, and selected keys
        coords_tensor, keys_list, indices_list, size_list, selected_keys = sample_subset(coords_tensor,
                                                                                         keys_list,
                                                                                         indices_list,
                                                                                         size_list,
                                                                                         key_mask,
                                                                                         ratio=args.ratio)
        # Add noise to the coordinates if specified
        if args.training.coordinate_noise > 0.0:
            coords_tensor = coords_tensor + (torch.rand_like(coords_tensor) - 0.5) * args.training.coordinate_noise
        # Sample the weights for the model
        model_cls, reconstructed_weights = sample_weights(model, model_cls,
                                                          coords_tensor, keys_list, indices_list, size_list, key_mask, selected_keys,
                                                          device=device, NORM=args.dimensions.norm)

        # Forward pass
        predict = model_cls(x)
        
        results=torch.argmax(predict,dim=1)
        train_acc=accuracy_score(results.cpu(), target.cpu())
        
        # Compute classification loss
        cls_loss = criterion(predict, target) 
        # Compute regularization loss
        reg_loss = sum([torch.norm(w, p=2) for w in reconstructed_weights])

        # Compute reconstruction loss if ground truth model is available
        if f"{hidden_dim}" in gt_model_dict:
            gt_model = gt_model_dict[f"{hidden_dim}"]
            gt_selected_weights = [
                w for k, w in gt_model.learnable_parameter.items() if k in selected_keys]

            reconstruct_loss = weighted_regression_loss(
                reconstructed_weights, gt_selected_weights)
        else:
            reconstruct_loss = torch.tensor(0.0)

        # Compute the total loss
        loss = args.hyper_model.loss_weight.ce_weight * cls_loss + args.hyper_model.loss_weight.reg_weight * \
            reg_loss + args.hyper_model.loss_weight.recon_weight * reconstruct_loss

        # Zero the gradients of the updated weights
        for updated_weight in model_cls.parameters():
            updated_weight.grad = None

        # Compute the gradients of the reconstructed weights
        loss.backward(retain_graph=True)
        torch.autograd.backward(reconstructed_weights, [
                                w.grad for k, w in model_cls.named_parameters() if k in selected_keys])

        # Clip the gradients if specified
        if args.training.get('clip_grad', 0.0) > 0:
            torch.nn.utils.clip_grad_value_(
                model.parameters(), args.training.clip_grad)

        # Update the weights
        optimizer.step()
        # Update the EMA if specified
        if ema:
            ema.update()  # Update the EMA after each training step
        total_loss += loss.item()

        # Update the AverageMeter objects
        losses.update(loss.item())
        cls_losses.update(cls_loss.item())
        reg_losses.update(reg_loss.item())
        reconstruct_losses.update(reconstruct_loss.item())

        # Log the losses and learning rate to wandb
        if batch_idx % args.experiment.log_interval == 0:
            wandb.log({
                "Running training accuracy argmax" : train_acc,
                "Running average training loss": losses.avg,
                "Cls Loss": cls_losses.avg,
                "Reg Loss": reg_losses.avg,
                "Reconstruct Loss": reconstruct_losses.avg,
                "Learning rate": optimizer.param_groups[0]['lr']
            }, step=batch_idx + epoch_idx * len(train_loader))
            # Print the losses and learning rate
            print(
                f"Iteration {batch_idx}: Loss = {losses.avg:.4f}, Reg Loss = {reg_losses.avg:.4f}, Reconstruct Loss = {reconstruct_losses.avg:.4f}, Cls Loss = {cls_losses.avg:.4f}, Learning rate = {optimizer.param_groups[0]['lr']:.4e}")
        
    tr_loss, tr_acc = validate_single(model_cls, train_loader, nn.CrossEntropyLoss(), args=args, device=device)
    wandb.log({
                "trainLoss_last model of the epoch" : tr_loss,
                "trainAcc_last model of the epoch" : tr_acc
                
            })
    return losses.avg, dim_dict, gt_model_dict

# Function to register hooks and print output shapes
def register_hooks_and_print_shapes(model, input_tensor):
    output_shapes = {}
    learnable_keys = set(model.learnable_parameter.keys())
    
    def hook_fnc(module_name):
        def hook_fn(module, input, output):
            class_name = module.__class__.__name__
            module_idx = len(output_shapes)
            m_key = f"{module_name}_{module_idx}_{class_name}"
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

# Function to initialize the model dictionary
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
        # Create a model for the given dimension
        model_cls = create_model(args.model.type, 
                                 hidden_dim=dim, 
                                 path=args.model.pretrained_path, 
                                 smooth=args.model.smooth).to(device)
        # Sample the coordinates, keys, indices, and size for the model
        coords_tensor, keys_list, indices_list, size_list = sample_coordinates(model_cls)
        # Add the model, coordinates, keys, indices, size, and key mask to the dictionary
        dim_dict[f"{dim}"] = (model_cls, coords_tensor, keys_list, indices_list, size_list, None)
        
        # Register hooks and print output shapes
        input_tensor = torch.randn(1, 3, 32, 32).to(device)
        #register_hooks_and_print_shapes(model_cls, input_tensor)
        
        
        # If the dimension is the starting dimension, add the ground truth model to the dictionary
        if dim == args.dimensions.start:
            print(f"Loading model for dim {dim}")
            model_trained = create_model(args.model.type, 
                                         hidden_dim=dim, 
                                         path=args.model.pretrained_path, 
                                         smooth=args.model.smooth).to(device)
            model_trained.eval()
            
            gt_model_dict[f"{dim}"] = model_trained
    return dim_dict, gt_model_dict

# Main function to train the model
def main():
    # Parse the arguments
    args = parse_args()

    # Print the arguments
    print_omegaconf(args)

    # Set the random seed
    set_seed(args.experiment.seed)

    # Get the training and validation data loaders
    train_loader, val_loader = get_cifar10(args.training.batch_size, 
                                           strong_transform=args.training.get('strong_aug', None))
    
    # Create the model for the starting dimension
    model = create_model(args.model.type, 
        hidden_dim=args.dimensions.start, 
        path=args.model.pretrained_path, 
        smooth=args.model.smooth, fuse=args.model.fuse).to(device)

    # Print the maximum dimension of the model
    print("Maximum DIM: ",find_max_dim(model))

    # Validate the model for the starting dimension
    val_loss, val_acc = validate_single(model, val_loader, nn.CrossEntropyLoss(), args=args, device=device)
    print(f"Initial Permutated model Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc*100:.2f}%")

    # Get the learnable parameters of the model
    checkpoint = model.learnable_parameter
    # Get the number of parameters
    number_param = len(checkpoint)
    # Print the keys of the parameters and the number of parameters
    print(f"Number of parameters to be learned: {number_param}")    
    print(f"Parameters keys: {model.keys}")

    # Get the hypermodel
    hyper_model = get_hypernet(args, number_param, device=device)
    # Initialize the EMA
    ema = EMA(hyper_model, decay=args.hyper_model.ema_decay)
    # Get the criterion, validation criterion, optimizer, and scheduler
    criterion, val_criterion, optimizer, scheduler = get_optimizer(args, hyper_model)

    # Initialize the starting epoch and best accuracy
    start_epoch = 0
    best_acc = 0.0
    
    # Create the directory to save the model
    os.makedirs(args.training.save_model_path, exist_ok=True)

    # If specified, load the checkpoint
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint_info, hyper_model = load_checkpoint(args.resume_from, hyper_model, optimizer, ema, device=device)
        start_epoch = checkpoint_info['epoch']
        best_acc = checkpoint_info['best_acc']
        print(f"Resuming from epoch: {start_epoch}, best accuracy: {best_acc*100:.2f}%")
        # Note: If there are more elements to retrieve, do so here.
    
    # If not testing, initialize wandb, the model dictionary, and the ground truth model dictionary
    if args.test == False:
        initialize_wandb(args)
        dim_dict, gt_model_dict = init_model_dict(args)
        dim_dict = shuffle_coordiates_all(dim_dict)
        
        # Iterate over the epochs
        for epoch in range(start_epoch, args.experiment.num_epochs):
            
            # Train the model for one epoch
            train_loss, dim_dict, gt_model_dict = train_one_epoch(hyper_model, train_loader, optimizer, criterion, dim_dict, gt_model_dict, epoch_idx=epoch, ema=ema, args=args)
            # Step the scheduler
            scheduler.step()

            # Print the training loss and learning rate
            print(f"Epoch [{epoch+1}/{args.experiment.num_epochs}], Training Loss: {train_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

            # If it's time to evaluate the model
            if (epoch + 1) % args.experiment.eval_interval == 0:
                # If EMA is specified, apply it
                if ema :
                    ema.apply()
                    
                # Sample the merged model
               # sampled_model = sample_merge_model(hyper_model, gt_model_dict[f"{args.dimensions.start}"], args, device=device)
                sampled_model = sample_single_model(hyper_model, gt_model_dict[f"{args.dimensions.start}"],cfg=args ,device=device)
                # Validate the merged model
                train_loss, train_acc = validate_single(sampled_model, train_loader, val_criterion, args=args, device=device)
                val_loss, val_acc = validate_single(sampled_model, val_loader, val_criterion, args=args, device=device)
                
                # If EMA is specified, restore the original weights
                if ema :
                    ema.restore()  # Restore the original weights
                    
                # Log the validation loss and accuracy to wandb
                wandb.log({
                    "Train Loss_model sampled outside training": train_loss,
                    "Train Accuracy_model sampled outside training": train_acc,
                    "Validation Loss_model sampled outside training": val_loss,
                    "Validation Accuracy_model sampled outside training": val_acc
                })
                # Print the validation loss and accuracy
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print(f"Epoch [{epoch+1}/{args.experiment.num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc*100:.2f}%")
                print(f"Epoch [{epoch+1}/{args.experiment.num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc*100:.2f}%")
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                
                # Save the checkpoint if the accuracy is better than the previous best
                if (val_acc > best_acc) and (epoch > 20):
                    best_acc = val_acc
                    save_checkpoint(f"{args.training.save_model_path}/cifar10_nerf_best.pth",hyper_model,optimizer,ema,epoch,best_acc)
                    print("------------------------------------------------------------------------------------------------------------------------------")
                    print(f"Checkpoint saved at epoch {epoch} with accuracy: {best_acc*100:.2f}%")
                    print("------------------------------------------------------------------------------------------------------------------------------")
        wandb.finish()
        
        print("Training finished.")
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        #testing the best model
        checkpoint_info, hyper_model = load_checkpoint(f"{args.training.save_model_path}/cifar10_nerf_best.pth", hyper_model, optimizer, ema, device=device)
        for hidden_dim in range(16, 81):
            # Create a model for the given hidden dimension
            model = create_model(args.model.type, 
                                    hidden_dim=hidden_dim, 
                                    path=args.model.pretrained_path, 
                                    smooth=args.model.smooth).to(device)

            # Sample the merged model for K times
            accumulated_model = sample_merge_model(hyper_model, model, args, K=100, device=device)

            # Validate the merged model
            val_loss, val_acc = validate_single(accumulated_model, val_loader, val_criterion, args=args, device=device)

            # Print the results
            print(f"Test using model {args.model}: hidden_dim {hidden_dim}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc*100:.2f}%")

    # If testing, iterate over the hidden dimensions and test the model
    else:
        for hidden_dim in range(16, 81):
            # Create a model for the given hidden dimension
            model = create_model(args.model.type, 
                                 hidden_dim=hidden_dim, 
                                 path=args.model.pretrained_path, 
                                 smooth=args.model.smooth,
                                 fuse=args.model.fuse).to(device)

            # If EMA is specified, apply it
            if ema:
                print("Applying EMA")
                ema.apply()
                
            # Sample the merged model
            accumulated_model = sample_merge_model(hyper_model, model, args, K=100, device=device)

            # Validate the merged model
            val_loss, val_acc = validate_single(accumulated_model, val_loader, val_criterion, args=args, device=device)
            
            # If EMA is specified, restore the original weights after applying EMA
            if ema:
                ema.restore()  # Restore the original weights after applying EMA
            
            # Save the model
            save_name = os.path.join(args.training.save_model_path, f"cifar10_{accumulated_model.__class__.__name__}_dim{hidden_dim}_single.pth")
            torch.save(accumulated_model.state_dict(), save_name)

            # Print the results
            print(f"Test using model {args.model}: hidden_dim {hidden_dim}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc*100:.2f}%")
            
            # Define the directory and filename structure
            filename = f"cifar10_results_{args.experiment.name}.txt"
            filepath = os.path.join(args.training.save_model_path, filename)

            # Write the results. 'a' is used to append the results; a new file will be created if it doesn't exist.
            with open(filepath, "a") as file:
                file.write(f"Hidden_dim: {hidden_dim}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc*100:.2f}%\n")
                # Print message
    
 
  
if __name__ == "__main__":
    main()