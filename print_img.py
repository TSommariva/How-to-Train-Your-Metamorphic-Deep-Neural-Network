import contextlib
import io
import numpy as np
import torch
from neumeta.models import create_model_cifar100 as create_model
import matplotlib.pyplot as plt
import os
import sys
import random
from neumeta.utils import (get_hypernet, sample_coordinates, sample_merge_model, shuffle_coordiates_all, validate_single, 
                           load_checkpoint, load_trained_blocks, parse_args, print_omegaconf, get_cifar100, DataLoader)


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
    for dim in range(1, 97):
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

def main(args):
    random_value = random.randint(0,1000000)
    
    #_, val_loader = get_cifar100(args.training.batch_size, num_workers=8)
    #trained_blocks = load_trained_blocks(args.resume_from)
    #dim_dict, gt_model_dict = init_model_dict(args, trained_blocks, args.model.single_block)
    #dim_dict = shuffle_coordiates_all(dim_dict)
    #_, _, _, keys_list, _, _, _ = dim_dict[f"{64}"]
    #selected_keys = np.unique(keys_list)
    #hyper_model = get_hypernet(args, 4 * trained_blocks, total_param=4 * trained_blocks ,key_list=selected_keys,device=device)
    #checkpoint_info, hyper_model, _, _, _ = load_checkpoint(args.resume_from, hyper_model, None, None, None, args=args)
    #
    #
    #start_epoch = checkpoint_info['epoch']
    #best_acc = checkpoint_info['best_acc']
    #start_block = trained_blocks
    #backbone_parameters = checkpoint_info['backbone_parameters']
    #print(f"Resuming from block: {start_block}, epoch: {start_epoch}, best accuracy: {best_acc*100:.2f}%")
    #
    #dimensions = torch.arange(4,97,4)
    #accuracy = torch.empty(24)
    #loss = torch.empty(24)
    #for i, hidden_dim in enumerate(range(4,97,4)):
    #    sampled_model = sample_merge_model(hyper_model, dim_dict[f"{hidden_dim}"][0], args, backbone_parameters=backbone_parameters ,device=device, K=100)
    #    val_loss, acc = validate_single(sampled_model, val_loader, torch.nn.CrossEntropyLoss(), args=args, device=device)
    #        
    #    accuracy[i] = acc * 100
    #    loss[i] = val_loss
    #    # Optionally, print the results to the console
    #    print(f"Hidden_dim: {hidden_dim}, Validation Loss: {val_loss:.4f}, Accuracy: {acc * 100:.2f}")
    plt.rcParams.update({
        'font.size': 8,         # Global default font size
        'axes.titlesize':  8,   # Axis title font size
        'axes.labelsize':  8,    # Axis label font size
        'xtick.labelsize': 8,   # X-tick label font size
        'ytick.labelsize': 8    # Y-tick label font size
    })
    # Create a 1x2 figure
    checkpoint = torch.load("/homes/tsommariva/neumeta/assets/accuracyVSdim.pth", weights_only=False)
    dimensions = checkpoint['dimensions']
    accuracy = checkpoint['accuracy']
    loss = checkpoint['loss']
    hist_w, hist_h = 6, 4    # inches

    # choose a width that suits your minipage (here: half of 6″ minus a bit of margin)
    subplot_w = hist_w / 2  # ≃3
    subplot_h = hist_h / 2 +0.35 # ≃2

    fig, (ax_acc, ax_loss) = plt.subplots(
        2, 1,
        figsize=(subplot_w, subplot_h),
        constrained_layout=True
    )

    # --- Accuracy subplot ---
    ax_acc.plot(dimensions, accuracy, color='#D23F0F',linewidth=1.5 ,label='Accuracy')
    ax_acc.set_xlabel('Hidden Dimension')
    ax_acc.set_ylabel('Accuracy (%)')
    ax_acc.axvspan(32, 64, color='gray', alpha=0.35)

    highlight_box = dict(facecolor='#F6CB52', edgecolor='none', boxstyle='round,pad=0.3', alpha=0.5)
    ax_acc.text(0.16, 0.2, 'Untrained', transform=ax_acc.transAxes,
                ha='center', va='center', fontweight='bold', color='black',
                bbox=highlight_box)
    ax_acc.text(0.82, 0.2, 'Untrained', transform=ax_acc.transAxes,
                ha='center', va='center', fontweight='bold', color='black',
                bbox=highlight_box)
    ax_acc.spines['top'].set_visible(False)
    ax_acc.spines['right'].set_visible(False)
    ax_acc.grid(True, linestyle=':')

    # --- Loss subplot ---
    ax_loss.plot(dimensions, loss, color='#D23F0F',linewidth=1.5, label='Loss')
    ax_loss.set_xlabel('Hidden Dimension')
    ax_loss.set_ylabel('Loss')
    ax_loss.axvspan(32, 64, color='gray', alpha=0.35)

    # Add the same text markers
    ax_loss.text(0.17, 0.8, 'Untrained', transform=ax_loss.transAxes,
                 ha='center', va='center', fontweight='bold', color='black',
                 bbox=highlight_box)
    ax_loss.text(0.82, 0.8, 'Untrained', transform=ax_loss.transAxes,
                 ha='center', va='center', fontweight='bold', color='black',
                 bbox=highlight_box)
    
    ax_loss.spines['top'].set_visible(False)
    ax_loss.spines['right'].set_visible(False)
    ax_loss.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(f'/homes/tsommariva/neumeta/assets/accuracyVSdim_{random_value}.png', dpi=1200, bbox_inches='tight')
    plt.savefig(f'/homes/tsommariva/neumeta/assets/accuracyVSdim_{random_value}.pdf', dpi=1200, bbox_inches='tight')
    #plt.show()
    print(f'accuracyVSdim_{random_value}')
    
    #checkpoint = {
    #    'dimensions': dimensions,
    #    'accuracy' : accuracy,
    #    'loss' : loss
    #}
    #torch.save(checkpoint,"/homes/tsommariva/neumeta/assets/accuracyVSdim.pth")


if __name__ == "__main__":
    args = parse_args()
    #print_omegaconf(args)
    main(args)
    print("Finished")