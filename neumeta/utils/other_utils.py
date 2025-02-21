import datetime
import torch
import numpy as np
import random
from prettytable import PrettyTable
from omegaconf import OmegaConf
import argparse
import torch.nn as nn
import torch.nn.functional as F
from neumeta.models import BasicBlock, BasicBlock_Resize
import wandb
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import  MultiStepLR
import copy
from neumeta.utils.hypernet_utils import get_hypernet


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a NeRF model with CIFAR-10"
    )

    parser.add_argument('--config', type=str, required=True,
                        help='Path to the configuration file')
    parser.add_argument('--ratio', type=float, default=1.0,
                        help='Ratio used for training purposes')
    parser.add_argument('--resume_from', type=str,
                        help='Checkpoint file path to resume training from')
    parser.add_argument('--load_from', type=str,
                        help='Checkpoint file path to load')
    parser.add_argument('--test_result_path', type=str,
                        help='Path to save the test result')
    parser.add_argument('--test', action='store_true',
                        default=False, help='Test the model')

    args, overrides = parser.parse_known_args()
    # Remove leading '--' if present in overrides
    overrides = [arg.lstrip('--') for arg in overrides]
    
    config = OmegaConf.load(args.config)

    if config.get('base_config', None):
        print("Loading base config from " + config.base_config)
        base_config = OmegaConf.load(config.base_config)
        config = OmegaConf.merge(base_config, config)

    cli_args = vars(args)
    overrides_args = OmegaConf.from_dotlist(overrides)

    config = OmegaConf.merge(config, OmegaConf.create(cli_args), overrides_args)

    return config


def print_omegaconf(cfg):
    """
    Print an OmegaConf configuration in a table format.

    :param cfg: OmegaConf configuration object.
    """
    # Flatten the OmegaConf configuration to a dictionary
    flat_config = OmegaConf.to_container(cfg, resolve=True)

    # Create a table with PrettyTable
    table = PrettyTable()

    # Define the column names
    table.field_names = ["Key", "Value"]

    # Recursively go through the items and add rows
    def add_items(items, parent_key=""):
        for k, v in items.items():
            current_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                # If the value is another dict, recursively add its items
                add_items(v, parent_key=current_key)
            else:
                # If it's a leaf node, add it to the table
                table.add_row([current_key, v])

    # Start adding items from the top-level configuration
    add_items(flat_config)

    # Print the table
    print(table)


def set_seed(seed_value=42):
    """Set the seed for generating random numbers for PyTorch and other libraries to ensure reproducibility.

    Args:
        seed_value (int, optional): The seed value. Defaults to 42 (a commonly used value in randomized algorithms requiring a seed).
    """
    print("Setting seed..." + str(seed_value) + " for reproducibility")
    # Set the seed for generating random numbers in Python's random library.
    random.seed(seed_value)

    # Set the seed for generating random numbers in NumPy, which can also affect randomness in cases where PyTorch relies on NumPy.
    np.random.seed(seed_value)

    # Set the seed for generating random numbers in PyTorch. This affects the randomness of various PyTorch functions and classes.
    torch.manual_seed(seed_value)

    # If you are using CUDA, and want to generate random numbers on the GPU, you need to set the seed for CUDA as well.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        # For multi-GPU, if you are using more than one GPU.
        torch.cuda.manual_seed_all(seed_value)

        # Additionally, for even more deterministic behavior, you might need to set the following environment, though it may slow down the performance.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        self.set_shadow(model)

    def set_shadow(self, model):
        # Initialize the shadow weights with the model's weights
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = param.clone()

    def apply(self):
        # Backup the current model weights and set the model's weights to the shadow weights
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.backup[name] = param
                    param.copy_(self.shadow[name])

    def restore(self):
        # Restore the original model weights
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.copy_(self.backup[name])

    def update(self):
        # Update the shadow weights
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param
                
    def extend(self, model):
        self.model = model
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name not in self.shadow:
                    self.shadow[name] = param.clone()

class EMA_ddp:
    def __init__(self, model, decay, rank):
        self.model = model
        self.decay = decay
        self.rank = rank
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = torch.zeros_like(param.data)
                self.backup[name] = torch.zeros_like(param.data)
                
        self.set_shadow(model)

    def set_shadow(self, model):
        # Initialize the shadow weights with the model's weights
        if self.rank == 0:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = param.data.clone()
        
        if torch.distributed.is_initialized():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    torch.distributed.broadcast(self.shadow[name], src=0)

    def apply(self):
        # Backup the current model weights and set the model's weights to the shadow weights
        if self.rank == 0: 
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    module_name = f"module.{name}" if not name.startswith('module.') else name
                    self.backup[module_name] = param.data.clone()
                    param.data = self.shadow[module_name]
        
        # Synchronize model parameters across ranks
        if torch.distributed.is_initialized():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    torch.distributed.broadcast(param.data, src=0)
                    torch.distributed.broadcast(self.backup[name], src=0)

    def restore(self):
        if self.rank == 0:
            # Restore weights only on rank 0
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    module_name = f"module.{name}" if not name.startswith('module.') else name
                    if module_name in self.backup:
                        param.data = self.backup[module_name]
                    else:
                        print(f"Warning: {module_name} not found in backup")

        # Sync restored weights across all ranks
        if torch.distributed.is_initialized():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    torch.distributed.broadcast(param.data, src=0)
                    
    def update(self):
        # Update the shadow weights
        if self.rank==0:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = self.decay * \
                        self.shadow[name] + (1.0 - self.decay) * param.data
        
        # Synchronize model parameters across ranks
        if torch.distributed.is_initialized():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    torch.distributed.broadcast(self.shadow[name], src=0)

def save_checkpoint(filepath, model, optimizer,scheduler ,ema, epoch, best_acc, backbone_parameters ,trained_blocks=1):
    """
    Saves the current state including a model, optimizer, and EMA shadow weights.

    Args:
    filepath (str): The file path where the checkpoint will be saved.
    model (torch.nn.Module): The model.
    optimizer (torch.optim.Optimizer): The optimizer.
    ema (EMA): The EMA object.
    epoch (int): The current epoch.
    best_acc (float): The best accuracy observed during training.
    """
    # Save the model, optimizer, EMA shadow weights, and other elements
    model.eval()
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict' : scheduler.state_dict(),
        'best_acc': best_acc,
        'trained_blocks': trained_blocks,
        'backbone_parameters': backbone_parameters
    }
    if ema:
        checkpoint['ema_shadow']=ema.shadow
    torch.save(checkpoint, filepath)
    
def save_checkpoint_ddp(filepath, model, optimizer,scheduler ,ema, epoch, best_acc ,trained_blocks=1):
    """
    Saves the current state including a model, optimizer, and EMA shadow weights.

    Args:
    filepath (str): The file path where the checkpoint will be saved.
    model (torch.nn.Module): The model.
    optimizer (torch.optim.Optimizer): The optimizer.
    ema (EMA): The EMA object.
    epoch (int): The current epoch.
    best_acc (float): The best accuracy observed during training.
    """
    # Save the model, optimizer, EMA shadow weights, and other elements
    if torch.distributed.get_rank() == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict' : scheduler.state_dict(),
            'best_acc': best_acc,
            'trained_blocks': trained_blocks,
        }
        if ema is not None:
            checkpoint['ema_shadow']=ema.shadow
        
        torch.save(checkpoint, filepath)

def load_trained_blocks(filepath):
    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
    
    return checkpoint['trained_blocks']
    
def load_checkpoint(filepath, model, optimizer, scheduler,ema, device='cuda', args=None):
    """
    Loads the state from a checkpoint into the model, optimizer, and EMA object.

    Args:
    filepath (str): The file path to load the checkpoint from.
    model (torch.nn.Module): The model.
    optimizer (torch.optim.Optimizer): The optimizer.
    ema (EMA): The EMA object.
    """
    try:
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
    except FileNotFoundError as e:
        print(f"Error: Could not find checkpoint file at {filepath}")
        print(f"Details: {str(e)}")
        return None, model, optimizer, scheduler, ema
    
    # After loading the checkpoint
    #saved_keys = set(checkpoint['model_state_dict'].keys())
    #model_keys = set(model.state_dict().keys())
    #print("Keys in saved state_dict but not in model:", saved_keys - model_keys)
    #print("Keys in model but not in saved state_dict:", model_keys - saved_keys)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except ValueError as e:
            print("ERROR!!")
            print(f"{e}")
            print("it's not possible to load the saved optimizer, a new one will be created instead")
            optimizer = None
            
    if 'scheduler_state_dict' in checkpoint and scheduler is not None and optimizer is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    if ema and 'ema_shadow' in checkpoint:
        ema.shadow = {k: checkpoint['ema_shadow'][k].to(
            device) for k in checkpoint['ema_shadow']}
    else:
        ema = None

    return checkpoint, model, optimizer, scheduler, ema  # Contains other information like epoch, best_acc

def load_non_ddp_checkpoint_to_ddp(filepath, ddp_model, optimizer, scheduler, ema, device='cuda'):
    """Load non-DDP checkpoint into DDP model"""
    checkpoint = torch.load(filepath, map_location='cpu')
    
    # Convert state dict for DDP
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        # Add 'module.' prefix for DDP
        new_state_dict[f'module.{k}'] = v
    
    # Load converted state dict
    ddp_model.load_state_dict(new_state_dict)
    ddp_model.to(device)
    
    # Load optimizer and scheduler states
    #if optimizer is not None:
    #    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #    for state in optimizer.state.values():
    #        for k, v in state.items():
    #            if isinstance(v, torch.Tensor):
    #                state[k] = v.to(device)
    #                
    #if scheduler is not None and 'scheduler_state_dict' in checkpoint:
    #    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    ## Load EMA if present
    #if ema is not None and 'ema_shadow' in checkpoint:
    #    ema.shadow = {k: v.to(device) for k, v in checkpoint['ema_shadow'].items()}
    
    return checkpoint, ddp_model

def load_checkpoint_ddp(filepath, model, optimizer, scheduler,ema, device='cuda', args=None):
    """
    Loads the state from a checkpoint into the model, optimizer, and EMA object.

    Args:
    filepath (str): The file path to load the checkpoint from.
    model (torch.nn.Module): The model.
    optimizer (torch.optim.Optimizer): The optimizer.
    ema (EMA): The EMA object.
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    
    # After loading the checkpoint
    #saved_keys = set(checkpoint['model_state_dict'].keys())
    #model_keys = set(model.state_dict().keys())
    #print("Keys in saved state_dict but not in model:", saved_keys - model_keys)
    #print("Keys in model but not in saved state_dict:", model_keys - saved_keys)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if ema is not None:
        ema.shadow = {k: checkpoint['ema_shadow'][k].to(
            device) for k in checkpoint['ema_shadow']}
    # ema.shadow = {k:checkpoint['ema_shadow'][k].to(device) for k in checkpoint['ema_shadow'] }  # specifically loading shadow weights

    return checkpoint, model  # Contains other information like epoch, best_acc

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
    
    wandb.init(project="ninr", name=run_name, config=dict(config), group='cifar100', dir='/work/tesi_tsommariva')

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
            #if 'layer3' in name:
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

def extend_nerf_compose(base_model, custom_init, args, number_param, total_param, key_list, device='cuda'):
    """
    Extends existing NeRF_ResMLP_Compose model with a new one, handling DDP models.

    Args:
        base_model: Base model (DDP or regular)
        extension_model: Model to extend with (DDP or regular)

    Returns:
        Extended model wrapped in DDP if input was DDP
    """
    extension_model = get_hypernet(args, number_param ,total_param=total_param ,key_list=key_list ,device=device)
    # Get underlying models if DDP
    base = base_model.module if isinstance(base_model, torch.nn.parallel.DistributedDataParallel) else base_model
    extension = extension_model.module if isinstance(extension_model, torch.nn.parallel.DistributedDataParallel) else extension_model
    
    base_checkpoint = {k:v for k, v in base_model.named_parameters()}
    last = None
    i=0
    
    with torch.no_grad():
        if custom_init:
            if isinstance(base_model.model, nn.ModuleList):
                last = nn.ModuleList([copy.deepcopy(m) for m in base_model.model[-4:]])
            elif isinstance(base_model.model, nn.ModuleDict):
                last = nn.ModuleList([copy.deepcopy(v) for _, v in list(base_model.model.items())[-4:]])

        last_params = []
        for name, param in last.named_parameters():
            last_params.append(param)
            
        for name, param in extension_model.named_parameters():
            if name in base_checkpoint:
                param.copy_(base_checkpoint[name])
            elif last is not None:
                if param.shape == last_params[i].shape:
                    param.copy_(last_params[i])
                else:
                    print(f"src:{name} and previous param have different shapes")
                i+=1
                
            
    # Re-wrap with DDP if input was DDP
    if isinstance(base_model, torch.nn.parallel.DistributedDataParallel):
        return torch.nn.parallel.DistributedDataParallel(
            base,
            device_ids=[torch.distributed.get_rank()],
            output_device=torch.distributed.get_rank()
        )
    return extension_model

def get_cifar_optimizer(args, model):
    alpha_params = [p for n, p in model.named_parameters() if 'alpha' in n]
    classifier_params = [p for n, p in model.named_parameters() if 'fc' in n]
    
    excluded_substrings = ("alpha", "fc")
    excluded_keys = set(model.learnable_parameter.keys())
    backbone_params = [
        p for n, p in model.named_parameters()
        if n not in excluded_keys and not any(s in n for s in excluded_substrings)
    ]
    optimizer_name = args.training.get('cls_optimizer', 'adamw')
    if optimizer_name == 'adamw':
        optimizer = AdamW([{'params': alpha_params},
                            #{'params': backbone_params, 'lr': args.training.backbone_learning_rate},
                            {'params': classifier_params, 'lr': args.training.cls_learning_rate}],
                             lr=args.training.alpha_learning_rate, 
                             weight_decay=args.training.cls_weight_decay)
    elif optimizer_name == 'adam':
        optimizer = Adam([{'params': alpha_params},
                           #{'params': backbone_params, 'lr': args.training.backbone_learning_rate},
                           {'params': classifier_params, 'lr': args.training.cls_learning_rate}],
                            lr=args.training.alpha_learning_rate, 
                            weight_decay=args.training.cls_weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD([{'params': alpha_params},
                                      #{'params': backbone_params, 'lr': args.training.backbone_learning_rate},
                                      {'params': classifier_params, 'lr': args.training.cls_learning_rate}],
                                       lr=args.training.alpha_learning_rate, 
                                       weight_decay=args.training.cls_weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    return optimizer


