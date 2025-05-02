import contextlib
import io
import numpy as np
import torch
import torch_pruning as tp
from neumeta.models import create_model_cifar100 as create_model
from torchvision import datasets, transforms
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import random
import sys
from neumeta.utils import (get_hypernet, sample_coordinates, sample_merge_model, shuffle_coordiates_all, validate_single, 
                           load_checkpoint, load_trained_blocks, parse_args, print_omegaconf, DataLoader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_model_dict(args, num_blocks = 1, single_block = False):
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
    for dim in range(1, args.dimensions.range[1] + 1):
        with contextlib.redirect_stdout(io.StringIO()):
            model_cls = create_model(args.model.type, 
                                 hidden_dim=dim, num_param=num_blocks, bottom_up=args.model.bottom_up,single_block=single_block ,
                                 path=args.model.pretrained_path, smooth=args.model.smooth, fuse=args.model.smooth, prior=False, config_args=args).to(device)
  
        
            
        coords_tensor, keys_list, indices_list, size_list = sample_coordinates(model_cls)
        dim_dict[f"{dim}"] = (model_cls, None ,coords_tensor, keys_list, indices_list, size_list, None)
        
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
                                 path=args.model.pretrained_path, smooth=args.model.smooth, fuse=args.model.smooth, config_args=args).to(device)
            
            if device=="cuda" and torch.backends.cudnn.version() >= 7603:
                model_trained.to(device, memory_format=torch.channels_last)  # Module parameters need to be channels last
            else:
                model_trained.to(device)
            model_trained.eval()
            
            gt_model_dict[f"{dim}"] = model_trained
    return dim_dict, gt_model_dict

def setup_evaluation_data(batch_size=128):
    """
    Prepare the CIFAR-10 dataset for model evaluation.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.4865, 0.4409],
                            std=[0.2673, 0.2564, 0.2761])
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.4865, 0.4409],
                            std=[0.2673, 0.2564, 0.2761])
    ])
    train_dataset = datasets.CIFAR100(root='./data', train=True, transform=transform_train, download=True)
    val_dataset = datasets.CIFAR100(root='./data', train=False, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    return val_loader, train_loader

def default_entry():
    return {'pruning_ratio': [], 'accuracy': []}

def main(args):
    example_inputs = torch.randn(1, 3, 32, 32).to(device)
    data = defaultdict(default_entry)

    N_batchs = 10
    val_loader, train_loader = setup_evaluation_data()
    #Importance criteria
    imp_dict = {
        'Random Pruning': tp.importance.RandomImportance(),
        'Hessian-based Pruning': tp.importance.HessianImportance(group_reduction='first'),
        'Taylor-based Pruning': tp.importance.TaylorImportance(group_reduction='first'),     
        'L1 Norm Pruning': tp.importance.MagnitudeImportance(p=1, group_reduction='first'),
        'L2 Norm Pruning': tp.importance.MagnitudeImportance(p=2, group_reduction="first"),   
    }
    model_name = 'ResNet20'
    target_layer = 'layer3.' #2.conv1'
    model_name = 'ResNet56'
    #Specific layer to be pruned with varying degrees
    target_layer = 'layer3.8.conv1'
    iterative_steps = 5
    for imp_name, imp in imp_dict.items():
        # Experiment with different pruning ratios for the specific layer
        for pruning_ratio in [i * 0.05 for i in range(16)]:  # Adjust the range/sequence as needed
            # Reset the model before each pruning experiment
            model = create_model(args.model.type, 
                                 hidden_dim=64,
                                 num_param=args.model.num_param,
                                 bottom_up=args.model.bottom_up,
                                 single_block=False,
                                 path="/homes/tsommariva/neumeta/neumeta/pretrained_models/cifar100_resnet56-f2eff4c8.pt", 
                                 smooth=False, fuse=False,
                                 prior=True, config_args=args, prune=True).to(device)
            # Define the pruning configuration
            pruning_config = {
                'ignored_layers': [],  # Layers to exclude from pruning
                'pruning_ratio_dict': {},  # Specific pruning ratios per layer
            }
            for layer_name, layer_module in model.named_modules():
                if layer_name.startswith(target_layer) and '3.0' not in layer_name and '3.8' not in layer_name and 'conv1' in layer_name:
                    pruning_config['pruning_ratio_dict'][layer_module] = pruning_ratio  # Set specific pruning ratio
                else:
                    pruning_config['pruning_ratio_dict'][layer_module] = 0  # No pruning for other layers
                if layer_name.startswith('fc'):
                    pruning_config['ignored_layers'].append(layer_module)  # Exclude the final classifier
            # Initialize the pruner
            pruner = tp.pruner.MetaPruner(
                model=model,
                example_inputs=example_inputs,
                importance=imp,
                iterative_steps=iterative_steps,
                **pruning_config
            )
            for i in range(iterative_steps):
                print(f"Pruning step {i+1}/{iterative_steps} with {imp_name} importance and {pruning_ratio * 100} pruning ratio:"  )
                if isinstance(imp, tp.importance.HessianImportance):
                    # loss = F.cross_entropy(model(images), targets)
                    for k, (imgs, lbls) in enumerate(train_loader):
                        if k>=N_batchs: break
                        imgs = imgs.cuda()
                        lbls = lbls.cuda()
                        output = model(imgs) 
                        # compute loss for each sample
                        loss = torch.nn.functional.cross_entropy(output, lbls, reduction='none')
                        imp.zero_grad() # clear accumulated gradients
                        for l in loss:
                            model.zero_grad() # clear gradients
                            l.backward(retain_graph=True) # simgle-sample gradient
                            imp.accumulate_grad(model) # accumulate g^2
                elif isinstance(imp, tp.importance.TaylorImportance):
                    # loss = F.cross_entropy(model(images), targets)
                    for k, (imgs, lbls) in enumerate(train_loader):
                        if k>=N_batchs: break
                        imgs = imgs.cuda()
                        lbls = lbls.cuda()
                        output = model(imgs)
                        loss = torch.nn.functional.cross_entropy(output, lbls)
                        loss.backward()
                
                # Execute the pruning
                pruner.step()
            # Evaluate and display the model performance after pruning
            print(f"\nEvaluating model with {target_layer} pruned at {pruning_ratio * 100}%:")
            model.zero_grad()  # Clear any cached gradients
            # print(model)
            
            val_loss, acc = validate_single(model, val_loader, torch.nn.CrossEntropyLoss(), device=device)
            
            # Calculate the number of resulting channels
            resulting_channels = int(64 * (1 - pruning_ratio))  # Assuming 64 is the original number of channels
            data[imp_name]['pruning_ratio'].append(pruning_ratio)
            data[imp_name]['accuracy'].append(acc * 100)
            # Optionally, print the results to the console
            print(f"Method: {imp_name}, Pruning ratio: {pruning_ratio:.2f}, Resulting Channels: {resulting_channels}, "
                  f"Validation Loss: {val_loss:.4f}, Accuracy: {acc * 100:.2f}")
            # exit()
    
    trained_blocks = load_trained_blocks(args.resume_from)
    dim_dict, gt_model_dict = init_model_dict(args, trained_blocks, args.model.single_block)
    dim_dict = shuffle_coordiates_all(dim_dict)
    _, _, _, keys_list, _, _, _ = dim_dict[f"{64}"]
    selected_keys = np.unique(keys_list)
    hyper_model = get_hypernet(args, 4 * trained_blocks, total_param=4 * trained_blocks ,key_list=selected_keys,device=device)
    checkpoint_info, hyper_model, _, _, _ = load_checkpoint(args.resume_from, hyper_model, None, None, None, args=args)
    
    
    start_epoch = checkpoint_info['epoch']
    best_acc = checkpoint_info['best_acc']
    start_block = trained_blocks
    backbone_parameters = checkpoint_info['backbone_parameters']
    print(f"Resuming from block: {start_block}, epoch: {start_epoch}, best accuracy: {best_acc*100:.2f}%")
    for pruning_ratio in [i * 0.05 for i in range(16)]:
        resulting_channels = int(64 * (1 - pruning_ratio))
        sampled_model = sample_merge_model(hyper_model, dim_dict[f"{resulting_channels}"][0], args, backbone_parameters=backbone_parameters ,device=device, K=100)
        val_loss, acc = validate_single(sampled_model, val_loader, torch.nn.CrossEntropyLoss(), args=args, device=device)
            
        data['Ours']['pruning_ratio'].append(pruning_ratio)
        data['Ours']['accuracy'].append(acc * 100)
        # Optionally, print the results to the console
        print(f"Method: Ours, Pruning ratio: {pruning_ratio:.2f}, Resulting Channels: {resulting_channels}, "
              f"Validation Loss: {val_loss:.4f}, Accuracy: {acc * 100:.2f}")
    
    #checkpoint = torch.load("/homes/tsommariva/neumeta/assets/pruning_data.pth")
    #data = checkpoint['data']

    # If needed, ensure the loaded object is a defaultdict with the correct default factory:
    if not isinstance(data, defaultdict):
        data = defaultdict(default_entry, data)
    
    
    #try:
    #    checkpoint = {
    #        'data': data,
    #    }
    #    torch.save(checkpoint,"/homes/tsommariva/neumeta/assets/pruning_data.pth")
    #except Exception as e:
    #    print("Error while saving checkpoint:", e)
    
    
    random_value = random.randint(0,1000000)
    markers = ['o', 's', '^', 'X', '<', 'D']
    palette = ['#E71E24','#E15840','#79AF5D','#E49343','#8776B6','#4196CB']  
    # Plotting
    plt.rcParams.update({
        'font.size': 10,         # Global default font size
        'axes.titlesize': 19,   # Axis title font size
        'axes.labelsize': 19,    # Axis label font size
        'xtick.labelsize': 15,   # X-tick label font size
        'ytick.labelsize': 15    # Y-tick label font size
    })
    plt.figure(figsize=(4, 2))
    fig, ax = plt.subplots()
    # Add vertical dotted line at pruning ratio 0.5
    for i, (method, values) in enumerate(data.items()):
        sorted_pairs = sorted(zip(values['pruning_ratio'], values['accuracy']))
        pruning_ratios, accuracies = zip(*sorted_pairs)
        plt.plot(pruning_ratios, accuracies, marker=markers[i % len(markers)],linewidth=3,markersize=10,markeredgecolor='white',color=palette[i%len(palette)] ,label=method)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(True, linestyle=':')
    plt.xlabel('Compression Ratio (%)')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'/homes/tsommariva/neumeta/assets/pruning_{random_value}.pdf', dpi=1200, bbox_inches='tight')
    print(f'pruning_{random_value}')
    
    
    
# Entry point of the script
if __name__ == "__main__":
    #args = parse_args()
    #print_omegaconf(args)
    main(None)
