import torch
import numpy as np
import random
import wandb
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import  MultiStepLR
import torch.nn.functional as F
from neumeta.hypermodel import NeRF_MLP_Compose, NeRF_ResMLP_Compose, NeRF_ResMLP_ComposeDict, NeRF_HierarcResMLP_ComposeDict, NeRF_HierarcResMLP_Compose
from tqdm import tqdm
import copy
from neumeta.utils.other_utils import AverageMeter
import gc

def weighted_regression_loss(reconstructed_weights, gt_selected_weights, epsilon=1e-6):
    """
    Calculate weighted regression loss, with each element in the ground truth weights
    contributing a weight proportional to its absolute value.

    Args:
    - reconstructed_weights (list of Tensors): List of reconstructed weight tensors.
    - gt_selected_weights (list of Tensors): List of ground truth weight tensors.

    Returns:
    - torch.Tensor: The mean of the weighted regression losses.
    """
    # Initialize an empty list to store individual losses
    losses = []

    # Iterate over the pairs of reconstructed and ground truth weights
    for w, w_gt in zip(reconstructed_weights, gt_selected_weights):
        # Calculate the weights as the absolute values of the ground truth weights
        element_weights = torch.abs(w_gt) + epsilon
        element_weights = element_weights / torch.max(element_weights)
        
        # Compute the MSE loss element-wise
        element_loss = (w - w_gt) ** 2
        
        # Apply the weights to the loss
        weighted_loss = element_loss * element_weights 
        
        # Sum the weighted loss for the current pair and add it to the list
        losses.append(weighted_loss.mean())

    # Calculate the mean of the weighted losses
    reconstruct_loss = torch.mean(torch.stack(losses))

    return reconstruct_loss

def get_optimizer_scaledFT(args, hyper_model, train_parameters, ft_parameters) :
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = LabelSmoothingCrossEntropy()
    val_criterion = torch.nn.CrossEntropyLoss()
    optimizer_name = args.training.get('optimizer', 'adamw')

    if optimizer_name == 'adamw':
        optimizer = AdamW([
                        {'params': train_parameters},
                        {'params': ft_parameters, 'lr': args.training.learning_rate * args.training.get('ft_scalinigFactor', 0.1)}],
                          
                          lr=args.training.learning_rate, 
                          weight_decay=args.training.weight_decay)
    elif optimizer_name == 'adam':
        optimizer = Adam([
                        {'params': train_parameters},
                        {'params': ft_parameters, 'lr': args.training.learning_rate * args.training.get('ft_scalinigFactor', 0.1)}], 
                         lr=args.training.learning_rate, 
                         weight_decay=args.training.weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD([
                        {'params': train_parameters},
                        {'params': ft_parameters, 'lr': args.training.learning_rate * args.training.get('ft_scalinigFactor', 0.1)}],
                                    lr=args.training.learning_rate, 
                                    momentum=args.training.get('momentum', 0.9),
                                    weight_decay=args.training.weight_decay)
    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop([
                        {'params': train_parameters},
                        {'params': ft_parameters, 'lr': args.training.learning_rate * args.training.get('ft_scalinigFactor', 0.1)}],
                                        lr=args.training.learning_rate, 
                                        momentum=args.training.get('momentum', 0.9),
                                        weight_decay=args.training.weight_decay)
    elif optimizer_name == 'adagrad':
        optimizer = torch.optim.Adagrad([
                        {'params': train_parameters},
                        {'params': ft_parameters, 'lr': args.training.learning_rate * args.training.get('ft_scalinigFactor', 0.1)}],
                                        lr=args.training.learning_rate, 
                                        weight_decay=args.training.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    scheduler_name = args.training.get('scheduler', 'multistep')
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.95)
    if scheduler_name == 'cosine':
        print("Using cosine scheduler, T_max:", args.training.T_max)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.training.T_max,eta_min=args.training.eta_min)
    elif scheduler_name == 'multistep':
        scheduler = MultiStepLR(optimizer,
                                milestones=args.training.get('lr_steps', [args.experiment.num_epochs]), 
                                gamma=0.1)
    elif scheduler_name == 'warmup_cosine':
        warmup_epochs = args.training.get('warmup_epochs', 5)
        
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-5, end_factor=1.0, total_iters=warmup_epochs)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.training.T_max - warmup_epochs),eta_min=args.training.eta_min)
        
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    return criterion, val_criterion, optimizer, scheduler


def get_optimizer(args, hyper_model, first_block = False):
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = LabelSmoothingCrossEntropy()
    val_criterion = torch.nn.CrossEntropyLoss()
    optimizer_name = args.training.get('optimizer', 'adamw')
    if optimizer_name == 'adamw':
        optimizer = AdamW(filter(lambda p: p.requires_grad, hyper_model.parameters()), 
                          lr=args.training.learning_rate, 
                          weight_decay=args.training.weight_decay)
    elif optimizer_name == 'adam':
        optimizer = Adam(hyper_model.parameters(), 
                         lr=args.training.learning_rate, 
                         weight_decay=args.training.weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(hyper_model.parameters(), 
                                    lr=args.training.learning_rate, 
                                    momentum=args.training.get('momentum', 0.9),
                                    weight_decay=args.training.weight_decay)
    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(hyper_model.parameters(), 
                                        lr=args.training.learning_rate, 
                                        momentum=args.training.get('momentum', 0.9),
                                        weight_decay=args.training.weight_decay)
    elif optimizer_name == 'adagrad':
        optimizer = torch.optim.Adagrad(hyper_model.parameters(), 
                                        lr=args.training.learning_rate, 
                                        weight_decay=args.training.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    scheduler_name = args.training.get('scheduler', 'multistep')
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.95)
    if scheduler_name == 'cosine' or (scheduler_name == 'warmup_cosine' and not first_block):
        print("Using cosine scheduler, T_max:", args.training.T_max)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.training.T_max,eta_min=args.training.eta_min)
    elif scheduler_name == 'multistep':
        scheduler = MultiStepLR(optimizer,
                                milestones=args.training.get('lr_steps', [args.experiment.num_epochs]), 
                                gamma=0.1)
    elif scheduler_name == 'warmup_cosine' and first_block:
        warmup_epochs = args.training.get('warmup_epochs', 20)
        
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-5, end_factor=1.0, total_iters=warmup_epochs)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.training.T_max - warmup_epochs),eta_min=args.training.eta_min)
        
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    #elif scheduler_name == 'warmup_cosine' and not first_block:
    #    print("Using cosine scheduler, T_max:", args.training.T_max)
    #    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.training.T_max,eta_min=args.training.eta_min)
    elif scheduler_name == 'warmup_const_cosine':
        warmup_epochs = args.training.get('warmup_epochs', 20)
        start_decay = args.training.get('start_decay', 50)
        total_epochs = args.experiment.num_epochs
        eta_min = args.training.learning_rate * 0.1
        
        # Linear warmup from ~0 to lr
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=1e-5,end_factor=1.0,total_iters=warmup_epochs)
        
        # Keep constant from warmup_epochs to start_decay
        if first_block:
            constant_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=(start_decay - warmup_epochs))
        else:
            constant_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=start_decay)
            
        # Cosine decay from start_decay to final epoch
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=(total_epochs - start_decay), eta_min=eta_min)
        
        # Combine the schedulers
        if first_block:
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,
                schedulers=[warmup_scheduler, constant_scheduler, cosine_scheduler],
                milestones=[warmup_epochs, start_decay])
        else:
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,
                schedulers=[constant_scheduler, cosine_scheduler],
                milestones=[start_decay])
            
    return criterion, val_criterion, optimizer, scheduler

def get_hypernet(args, number_param, total_param = 32 ,key_list = None,device='cuda'):
    """
    Returns a hypernetwork model based on the specified hyper_model_type in the arguments.

    Args:
        args (argparse.Namespace): Arguments containing hyper_model_type and hyperparameters.
        number_param (int): Number of parameters to be generated by the hypernetwork.

    Returns:
        torch.nn.Module: Hypernetwork model based on the specified hyper_model_type.
    """
    hyper_model_type = args.hyper_model.get('type', 'mlp')
    print("Hyper model type: " + hyper_model_type)
    output_dim = args.hyper_model.output_dim
    if hyper_model_type == 'mlp':
        hyper_model = NeRF_MLP_Compose(
            input_dim=args.hyper_model.input_dim,
            hidden_dim=args.hyper_model.hidden_dim,
            num_layers=args.hyper_model.num_layers,
            output_dim=output_dim,
            num_freqs=args.hyper_model.num_freqs,
            num_compose=number_param,
            normalizing_factor=args.dimensions.norm,
            total_param = total_param,
            coordinate_noise = args.training.coordinate_noise
        ).to(device)
        
    elif hyper_model_type == 'resmlp':
        print("Using scalar", args.hyper_model.get('scalar', 0.1))
        hyper_model = NeRF_ResMLP_Compose(
            input_dim=args.hyper_model.input_dim,
            hidden_dim=args.hyper_model.hidden_dim,
            num_layers=args.hyper_model.num_layers,
            output_dim=output_dim,
            num_freqs=args.hyper_model.num_freqs,
            scalar=args.hyper_model.get('scalar', 0.1),
            num_compose=number_param,
            normalizing_factor=args.dimensions.norm,
            total_param = total_param,
            coordinate_noise = args.training.coordinate_noise
        ).to(device)
    elif hyper_model_type == 'resmlpDict':
        if key_list is None and number_param > 0:
            print("You need to specify a key_list in order to use:",hyper_model_type)
            return None
        print("Using scalar", args.hyper_model.get('scalar', 0.1))
        hyper_model = NeRF_ResMLP_ComposeDict(
            key_list=key_list,
            input_dim=args.hyper_model.input_dim,
            hidden_dim=args.hyper_model.hidden_dim,
            num_layers=args.hyper_model.num_layers,
            output_dim=args.hyper_model.output_dim,
            num_freqs=args.hyper_model.num_freqs,
            scalar=args.hyper_model.get('scalar', 0.1),
            normalizing_factor=args.dimensions.norm,
            total_param = total_param,
            coordinate_noise = args.training.coordinate_noise
        ).to(device)
    elif hyper_model_type == 'hierarchical_resmlpDict':
        if key_list is None and number_param > 0:
            print("You need to specify a key_list in order to use:",hyper_model_type)
            return None
        hyper_model = NeRF_HierarcResMLP_ComposeDict(
            key_list=key_list,
            num_kernel_groups=args.hyper_model.get('kernel_groups', 4),
            input_dim=args.hyper_model.input_dim,
            hidden_dim=args.hyper_model.hidden_dim,
            num_layers=args.hyper_model.num_layers,
            output_dim=args.hyper_model.output_dim,
            num_freqs=args.hyper_model.num_freqs,
            scalar=args.hyper_model.get('scalar', 0.1),
            normalizing_factor=args.dimensions.norm,
            total_param = total_param,
            coordinate_noise = args.training.coordinate_noise
            ).to(device)
    elif hyper_model_type == 'hierarchical_resmlp':
        print("hierarchical residual mlp, ",args.hyper_model.get('kernel_groups', 4), "kernel groups, using scalar: ",args.hyper_model.get('scalar', 0.1))
        hyper_model = NeRF_HierarcResMLP_Compose(
            input_dim=args.hyper_model.input_dim,
            hidden_dim=args.hyper_model.hidden_dim,
            num_layers=args.hyper_model.num_layers,
            output_dim=output_dim,
            num_freqs=args.hyper_model.num_freqs,
            scalar=args.hyper_model.get('scalar', 0.1),
            num_compose=number_param,
            num_kernel_groups=args.hyper_model.get('kernel_groups', 4),
            normalizing_factor=args.dimensions.norm,
            total_param = total_param,
            coordinate_noise = args.training.coordinate_noise
        ).to(device)
    else:
        raise ValueError(f"Unsupported hyper_model_type: {hyper_model_type}")
    
    if args.model.only_last and isinstance(hyper_model.model, torch.nn.ModuleDict):
        for module_key, module in hyper_model.model.items():
            if f"layer3_{number_param//4}" in module_key:
                for param in module.parameters():
                    param.requires_grad = True
            else:
                for param in module.parameters():
                    param.requires_grad = False
        
    return hyper_model

def validate_single(model_cls, val_loader, criterion, args=None, device='cuda'):
    val_loss = 0.0
    correct = 0
    total = 0
    preds = []
    gt = []
    model_cls = model_cls.to(device)
    model_cls.eval()
    
    with torch.no_grad():
        with tqdm(val_loader, miniters=23) as t_loader:
            for x, target in t_loader:
                if device=="cuda" and torch.backends.cudnn.version() >= 7603:
                    x, target = x.to(device, memory_format=torch.channels_last), target.to(device)
                else:
                    x, target = x.to(device), target.to(device)
                
                predict = model_cls(x)
    
                pred = torch.argmax(predict, dim=-1)
                correct += (pred == target).sum().item()
                total += target.size(0)
    
                loss = criterion(predict, target)
                val_loss += loss.item()
    
    accuracy = correct / total if total > 0 else 0
    return val_loss / len(val_loader), accuracy

def validate_all_dimensions(hypermodel,backbone_parameters, num_param ,val_loader, criterion, create_model ,args, device='cuda'):
    losses = AverageMeter()
    accuracies = []
    for hidden_dim in range(args.dimensions.range[0], args.dimensions.range[1]):
        model = create_model(args.model.type, 
                                hidden_dim=hidden_dim,
                                num_param=num_param,
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
        accumulated_model = sample_merge_model(hypermodel, model, args,backbone_parameters=backbone_parameters ,K=100, device=device)
        val_loss, val_acc = validate_single(accumulated_model, val_loader, criterion, args, device=device)
        print(f"Dimension:{hidden_dim} Validation loss:{val_loss:.4f} Validation Accuracy:{val_acc*100:.2f}")
        losses.update(val_loss)
        accuracies.append(val_acc)
        del model
        del accumulated_model
        gc.collect()
        torch.cuda.empty_cache()
    
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    wandb.log({"Non-prior Seen - Validation loss": losses.avg, "Non-prior Seen - Validation Accuracy": mean_accuracy*100}, commit=False)
    print(f"Non-prior Seen - Validation loss:{losses.avg:.4f} Non-prior Seen - Validation Accuracy:{mean_accuracy*100:.2f} ± {std_accuracy*100:.2f}")
    print("------------------------------------------------------------------------------------------------------------------------------")
    
    losses.reset()
    accuracies = []
    
    for hidden_dim in range(args.dimensions.range[0]//4 , args.dimensions.range[0]):
        model = create_model(args.model.type, 
                                hidden_dim=hidden_dim,
                                num_param=num_param,
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
        accumulated_model = sample_merge_model(hypermodel, model, args,backbone_parameters=backbone_parameters ,K=100, device=device)
        val_loss, val_acc = validate_single(accumulated_model, val_loader, criterion, args, device=device)
        print(f"Dimension:{hidden_dim} Validation loss:{val_loss:.4f} Validation Accuracy:{val_acc*100:.2f}")
        losses.update(val_loss)
        accuracies.append(val_acc)
        del model
        del accumulated_model
        gc.collect()
        torch.cuda.empty_cache()
    
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    wandb.log({"Non-prior Unseen Low - Validation loss": losses.avg, "Non-prior Unseen Low - Validation Accuracy": mean_accuracy*100}, commit=False)
    print(f"Non-prior Unseen Low - Validation loss:{losses.avg:.4f} Non-prior Unseen Low - Validation Accuracy:{mean_accuracy*100:.2f} ± {std_accuracy*100:.2f}")
    print("------------------------------------------------------------------------------------------------------------------------------")
    
    losses.reset()
    accuracies = []
    
    for hidden_dim in range(args.dimensions.range[1] + 1 , args.dimensions.range[1] + args.dimensions.range[0]//2 + 1):
        model = create_model(args.model.type, 
                                hidden_dim=hidden_dim,
                                num_param=num_param,
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
        accumulated_model = sample_merge_model(hypermodel, model, args,backbone_parameters=backbone_parameters ,K=100, device=device)
        val_loss, val_acc = validate_single(accumulated_model, val_loader, criterion, args, device=device)
        print(f"Dimension:{hidden_dim} Validation loss:{val_loss:.4f} Validation Accuracy:{val_acc*100:.2f}")
        losses.update(val_loss)
        accuracies.append(val_acc)
        del model
        del accumulated_model
        gc.collect()
        torch.cuda.empty_cache()
    
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    wandb.log({"Non-prior Unseen High - Validation loss": losses.avg, "Non-prior Unseen High - Validation Accuracy": mean_accuracy*100}, commit=False)
    print(f"Non-prior Unseen High - Validation loss:{losses.avg:.4f} Non-prior Unseen High - Validation Accuracy:{mean_accuracy*100:.2f} ± {std_accuracy*100:.2f}")
    print("------------------------------------------------------------------------------------------------------------------------------")
        
    return
    
def average_models(models):
    """
    Average the weights of multiple PyTorch models.
    
    Args:
    models (list of torch.nn.Module): List of models with the same architecture.
    
    Returns:
    torch.nn.Module: A model with the average weights.
    """
    # Validate if models list is not empty
    if not models:
        raise ValueError("No models to average.")
    
    # Create a copy of the first model. We will use this model to store the averaged weights
    averaged_model = copy.deepcopy(models[0])

    # Initialize a dict to hold the sum of all model parameters
    param_sum = dict()
    
    with torch.no_grad():
        for model in models:
            for name, param in model.named_parameters():
                if name not in param_sum:
                    # Initialize the sum for this parameter as a tensor filled with zeros with the same shape as the parameter
                    param_sum[name] = torch.zeros_like(param)

                # Add the parameters
                param_sum[name] += param

        # Average the sum of parameters by the number of models
        for name in param_sum:
            param_sum[name] = param_sum[name] / len(models)

        # Update the averaged model with the new averaged weights
        for name, param in averaged_model.named_parameters():
            param.copy_(param_sum[name].clone())
    
    return averaged_model

def sample_merge_model(hyper_model, model, args, backbone_parameters ,K=50, device='cuda'):
    # Initialize a model to accumulate the weights over K samples
    if isinstance(hyper_model, torch.nn.parallel.DistributedDataParallel):
        hyper_model = hyper_model.module
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in backbone_parameters: 
                param.copy_(backbone_parameters[name])
                    
    hyper_model.eval()
    model_cls_temp = copy.deepcopy(model)
    model_cls_temp.eval()
    param_average = dict()
    
    for name, param in model_cls_temp.named_parameters():
        param_average[name] = torch.zeros_like(param)
        
    for k in range(K):
        # Sampling and averaging weights
        coords_tensor, keys_list, indices_list, size_list = sample_coordinates(model_cls_temp)
        key_mask = create_key_masks(keys_list=keys_list)
        model_cls_temp, _ = sample_weights(hyper_model, model_cls_temp, coords_tensor, keys_list, indices_list, size_list, key_mask, list(key_mask.keys()), device=device, NORM=args.dimensions.norm)

        for name, param in model_cls_temp.named_parameters():
            param_average[name] += param/K

    for name, param in model_cls_temp.named_parameters():
        param.copy_(param_average[name])
        
    return model_cls_temp

def sample_coordinates(model_cls):
    """
    Sample coordinates for the given model_cls.

    Args:
        model_cls: The model class to sample coordinates for.

    Returns:
        A tuple containing:
        - coords_tensor: A tensor containing the sampled coordinates.
        - keys_list: A list of keys corresponding to each coordinate.
        - indices_list: A list of in_channel and out_channel indices.
    """
    if isinstance(model_cls, torch.nn.parallel.DistributedDataParallel):
        model_cls = model_cls.module
        
    checkpoint = model_cls.learnable_parameter
    
    coords_list = []  # List to store all coordinates
    keys_list = []  # List to store keys corresponding to each coordinate
    indices_list = []  # List to store in_channel and out_channel indices
    size_list = []  # List to store the size of each tensor
    
    layer_num = len(checkpoint)
    # Iterate over the new model's weights
    for i, (k, tensor) in enumerate(checkpoint.items()):
        
        # Handle 2D tensors (e.g., weight matrices)
        if len(tensor.shape) == 4:
            for in_channel in range(tensor.shape[0]):
                for out_channel in range(tensor.shape[1]):
                    coords = [i, in_channel, out_channel, layer_num, tensor.shape[0], tensor.shape[1]]
                    coords_list.append(coords)
                    keys_list.append(k)
                    indices_list.append((in_channel, out_channel, tensor.shape[2], tensor.shape[3]))
                    size_list.append(4)
                    
        elif len(tensor.shape) == 2:
            for in_channel in range(tensor.shape[0]):
                for out_channel in range(tensor.shape[1]):
                    coords = [i, in_channel, out_channel, layer_num, tensor.shape[0], tensor.shape[1]]
                    coords_list.append(coords)
                    keys_list.append(k)
                    indices_list.append((in_channel, out_channel, 0, 0))
                    size_list.append(2)
                    
        # Handle 1D tensors (e.g., biases)
        elif len(tensor.shape) == 1:
            for in_channel in range(tensor.shape[0]):
                coords = [i, in_channel, 0, layer_num, tensor.shape[0], 1]
                coords_list.append(coords)
                keys_list.append(k)
                indices_list.append((in_channel, 0, 0, 0))
                size_list.append(1)
                
    # Convert list of coordinates to tensor
    coords_tensor = torch.tensor(coords_list, dtype=torch.float32)
    indices_list = torch.tensor(indices_list, dtype=torch.int64)
    size_list = torch.tensor(size_list, dtype=torch.int64)
    keys_list = np.array(keys_list)
    return coords_tensor, keys_list, indices_list, size_list

def sample_single_model(hyper_model, model, device='cuda', cfg=None):
    # Initialize a model to accumulate the weights over K samples
    coords_tensor, keys_list, indices_list, size_list = sample_coordinates(model)
    key_mask = create_key_masks(keys_list=keys_list)
    with torch.no_grad():
        model, _ = sample_weights(hyper_model, model, coords_tensor, keys_list, indices_list, size_list, key_mask, list(key_mask.keys()), device=device, NORM=cfg.dimensions.norm)
    model.eval()
    return model

def sample_weights_Dict(model, model_cls, coords_tensor, keys_list, indices_list, size_list, key_mask, selected_keys=None, device='cuda',large_batch_size = 4096, NORM=1, scaler = None, block_flags = None):
    if selected_keys is not None:
        predicted_checkpoint = {k:v for k, v in model_cls.learnable_parameter.items() if k in selected_keys}
    else:
        predicted_checkpoint = model_cls.learnable_parameter
    
    # Sample a batch of coordinates
    coords_tensor = coords_tensor.to(device)
    input_tensor = (coords_tensor) # / NORM) 
    
    selected_mask = sum([key_mask[k] for k in selected_keys]).bool()
    
    # Iterate over the keys that have been selected for processing.
    for key in selected_keys:
        split_key = key.split('.')
        i_value = int(split_key[1])
        # Create a boolean mask based on the selected mask from the key_mask dictionary.
        boolean_mask = key_mask[key][selected_mask].bool()
        
        if block_flags is not None:
            if block_flags[i_value -1]:
                predicted_weights = model(input_tensor[boolean_mask], key)
            else:
                with torch.no_grad():
                    predicted_weights = model(input_tensor[boolean_mask], key)
        else:
            predicted_weights = model(input_tensor[boolean_mask], key)

        # Check the size information for the current mask and proceed accordingly.
        if size_list[boolean_mask][0] == 4:  # Condition for conv weights.
            # Extract height and width from the indices list.
            height, width = indices_list[boolean_mask][0, 2:]
            
            # Extract the relevant weights based on the mask.
            total_weights = predicted_weights.size(-1)
            
            # Adjust weights if they don't match the expected size (h*w).
            if height * width < total_weights:
                start_index = torch.div(total_weights, 2, rounding_mode='trunc') - torch.div(height * width, 2, rounding_mode='trunc')
                end_index = start_index + height * width
                predicted_weights = predicted_weights[:, start_index:end_index]
            
            # Reshape and assign the adjusted weights to the appropriate position in the checkpoint dictionary.
            predicted_checkpoint[key][indices_list[boolean_mask][:, 0], indices_list[boolean_mask][:, 1]].copy_(predicted_weights.view(-1 ,height, width))
        
        elif size_list[boolean_mask][0] == 1:  # Condition for conv biases.
            # Assign the weights to the specified indices.
            predicted_checkpoint[key][indices_list[boolean_mask][:, 0]].copy_(predicted_weights.view(-1))
        
        elif size_list[boolean_mask][0] == 2:  # Condition for a different size.
            # Directly assign the weights without reshaping.
            predicted_checkpoint[key][indices_list[boolean_mask][:, 0], indices_list[boolean_mask][:, 1]].copy_(predicted_weights[boolean_mask][:, 0])
         
    with torch.no_grad():     
        for name, param in model_cls.learnable_parameter.items():
            if name in predicted_checkpoint:
                param.copy_(predicted_checkpoint[name])

    return model_cls, predicted_checkpoint


def sample_weights(model, model_cls, coords_tensor, keys_list, indices_list, size_list, key_mask, selected_keys=None, device='cuda',large_batch_size = 4096, NORM=1, scaler = None, block_flags = None):
    """
    Samples weights from the model and updates the predicted_checkpoint using the batch of predicted weights.

    Args:
        model (nn.Module): The neural network model.
        model_cls (nn.Module): The neural network model class.
        coords_tensor (torch.Tensor): The coordinates tensor.
        keys_list (list): The list of keys.
        indices_list (list): The list of indices.
        selected_keys (list, optional): The list of selected keys. Defaults to None.
        device (str, optional): The device to use. Defaults to 'cuda'.
        large_batch_size (int, optional): The large batch size. Defaults to 4096.

    Returns:
        tuple: A tuple containing the updated model_cls and the list of predicted weights.
    """
    #if isinstance(model, torch.nn.parallel.DistributedDataParallel):
    #    model = model.module
    #if isinstance(model_cls, torch.nn.parallel.DistributedDataParallel):
    #    model_cls = model_cls.module
    
    if isinstance (model.model, torch.nn.ModuleDict):
        return sample_weights_Dict(model, model_cls, coords_tensor, keys_list, indices_list, size_list, key_mask, selected_keys, device,large_batch_size, NORM, scaler, block_flags)
    
    if selected_keys is not None:
        predicted_checkpoint = {k:v for k, v in model_cls.learnable_parameter.items() if k in selected_keys}
    else:
        predicted_checkpoint = model_cls.learnable_parameter
    
    # Sample a batch of coordinates
    coords_tensor = coords_tensor.to(device)
    layer_id = coords_tensor[:, 0].int()
    input_dim = coords_tensor[:, -1]
    input_tensor = (coords_tensor) #/NORM) 
    predicted_weights = model(input_tensor, layer_id=layer_id, input_dim=input_dim)
    
    selected_mask = sum([key_mask[k] for k in selected_keys]).bool()
    
    # Iterate over the keys that have been selected for processing.
    for key in selected_keys:
        # Create a boolean mask based on the selected mask from the key_mask dictionary.
        boolean_mask = key_mask[key][selected_mask].bool()

        # Check the size information for the current mask and proceed accordingly.
        if size_list[boolean_mask][0] == 4:  # Condition for conv weights.
            # Extract height and width from the indices list.
            height, width = indices_list[boolean_mask][0, 2:]
            
            # Extract the relevant weights based on the mask.
            current_weights = predicted_weights[boolean_mask]
            total_weights = current_weights.size(-1)
            
            # Adjust weights if they don't match the expected size (h*w).
            if height * width < total_weights:
                start_index = torch.div(total_weights, 2, rounding_mode='trunc') - torch.div(height * width, 2, rounding_mode='trunc')
                end_index = start_index + height * width
                current_weights = current_weights[:, start_index:end_index]
            
            # Reshape and assign the adjusted weights to the appropriate position in the checkpoint dictionary.
            predicted_checkpoint[key][indices_list[boolean_mask][:, 0], indices_list[boolean_mask][:, 1]] = current_weights.view(-1, height, width)
        
        elif size_list[boolean_mask][0] == 1:  # Condition for conv biases.
            # Assign the weights to the specified indices.
            predicted_checkpoint[key][indices_list[boolean_mask][:, 0]] = predicted_weights[boolean_mask][:, 0]
        
        elif size_list[boolean_mask][0] == 2:  # Condition for a different size.
            # Directly assign the weights without reshaping.
            predicted_checkpoint[key][indices_list[boolean_mask][:, 0], indices_list[boolean_mask][:, 1]] = predicted_weights[boolean_mask][:, 0]
    
    with torch.no_grad():     
        for name, param in model_cls.learnable_parameter.items():
            if name in predicted_checkpoint:
                param.copy_(predicted_checkpoint[name].clone())

    return model_cls, list(predicted_checkpoint.values())

def sample_subset(coords_tensor, keys_list, indices_list, size_list, key_mask, ratio=0.5):
    """
    Samples a subset of the input data based on the given ratio.

    Args:
        coords_tensor (numpy.ndarray): The coordinates tensor.
        keys_list (list): The list of keys.
        indices_list (list): The list of indices.
        ratio (float): The ratio of the subset to the original data.

    Returns:
        tuple: A tuple containing the subset coordinates tensor, the subset keys list,
            the subset indices list, and the selected keys list.
    """
    if ratio >= 1.0:
        return coords_tensor, keys_list, indices_list, size_list, np.unique(keys_list)
    assert len(coords_tensor) == len(keys_list) == len(indices_list)
    
    
    uniqe_keys = np.unique(keys_list)
    num_samples = int(len(uniqe_keys) * ratio)
    selected_keys = random.sample(list(uniqe_keys), k=num_samples)
    
    if key_mask is not None:
        selected_mask = sum([key_mask[k] for k in selected_keys]).bool()
    
    subset_coords = coords_tensor[selected_mask]
    subset_keys = keys_list[selected_mask]
    subset_indices = indices_list[selected_mask]
    subset_size = size_list[selected_mask]
    
    return subset_coords, subset_keys, subset_indices, subset_size, selected_keys

def shuffle_coordiates_all(dim_dict):
    """
    Shuffle the coordinates of all dimensions in the given range list.

    Args:
        dim_dict (dict): A dictionary containing the coordinates, keys, and indices for each dimension.
        range_list (list): A list of dimension indices to shuffle.

    Returns:
        dict: The updated dictionary with shuffled coordinates, keys, and indices for the specified dimensions.
    """
    for dim in dim_dict:
        (model_cls, cls_optimizer , coords_tensor, keys_list, indices_list, size_list, key_mask) = dim_dict[dim]
        # coords_tensor, keys_list, indices_list, size_list = shuffle_coordiates(coords_tensor, keys_list, indices_list, size_list)
        key_mask = create_key_masks(keys_list)
        dim_dict[dim] = (model_cls, cls_optimizer ,coords_tensor, keys_list, indices_list, size_list, key_mask)
        
    return dim_dict

def create_key_masks(keys_list):
    """
    Creates a dictionary of key masks for the given list of keys.

    Args:
    - keys_list (list): A list of keys for which to create masks.

    Returns:
    - key_mask_dict (dict): A dictionary of key masks, where each key corresponds to a tensor of zeros and ones.
    """
    unique_keys, key_inverse = np.unique(keys_list, return_inverse=True)
    key_mask_dict = {key: torch.tensor(key_inverse == i, dtype=torch.long) for i, key in enumerate(unique_keys)}

    return key_mask_dict

def shuffle_coordiates(coords_tensor, keys_list, indices_list, size_list):
    """
    Shuffle the coordinates tensor, keys list, and indices list randomly.

    Args:
        coords_tensor (numpy.ndarray): The coordinates tensor to shuffle.
        keys_list (list): The list of keys to shuffle in the same order as the coordinates tensor.
        indices_list (list): The list of indices to shuffle in the same order as the coordinates tensor.

    Returns:
        numpy.ndarray: The shuffled coordinates tensor.
        list: The shuffled list of keys.
        list: The shuffled list of indices.
    """
    # Shuffle data
    shuffled_indices = np.arange(len(coords_tensor))
    np.random.shuffle(shuffled_indices)
    coords_tensor = coords_tensor[shuffled_indices]
    keys_list = keys_list[shuffled_indices]
    indices_list = indices_list[shuffled_indices]
    size_list = size_list[shuffled_indices]
    return coords_tensor, keys_list, indices_list, size_list
