#from .resnet import ResNet18_convbn
# from .alexnet import AlexNet
from .resnet_imagenet import resnet18_imagenet
from .utils import fuse_module
from .resnet_cifar import *
from .lenet import MnistNet, MnistResNet
import torch
from smooth.permute import PermutationManager, compute_tv_loss_for_network
#from .resnet_tinyimagenet import resnet18_tinyimagenet
import os

def create_mnist_model(model_name, hidden_dim, depths=None, path=None):
    if model_name == "LeNet":
        model = MnistNet(hidden_dim=hidden_dim)
    elif model_name == "ResNet":
        model = MnistResNet(hidden_dim=hidden_dim, num_blocks=depths)
    elif model_name == "ResNet_width":
        model = MnistResNet(hidden_dim=hidden_dim, num_blocks=depths)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model
        
def create_model_cifar10(model_name, hidden_dim, path=None, smooth=False, fuse = True):
    """
    Create a model based on the specified name.

    :param model_name: String that specifies the model to use.
    :param path: Optional path for the model's weights.
    :return: The initialized model.
    """
    if model_name == "ResNet20":  # Add other models as you support them
        model = cifar10_resnet20(hidden_dim=hidden_dim)
    elif model_name == "ResNet32":  # Add other models as you support them
        model = cifar10_resnet32(hidden_dim=hidden_dim)
    elif model_name == "ResNet44":  # Add other models as you support them
        model = cifar10_resnet44(hidden_dim=hidden_dim)
    elif model_name == "ResNet56":  # Add other models as you support them
        model = cifar10_resnet56(hidden_dim=hidden_dim)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    if path and not smooth and not fuse:
        if os.path.exists(path):
            #model = torch.load(path,weights_only=False)
            fuse_module(model)
            state_dict = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
            load_checkpoint(model, state_dict)
            return model
     
    if path:
        if os.path.exists(path):
            print("Loading model from", path)
            state_dict = torch.load(path, map_location=torch.device('cpu'))
            load_checkpoint(model, state_dict)
    
    if fuse:
        fuse_module(model)
        
    if smooth:
        print("Smooth the parameters of the model")
        print("TV original model: ", compute_tv_loss_for_network(model, lambda_tv=1.0).item())
        input_tensor = torch.randn(1, 3, 32, 32)
        permute_func = PermutationManager(model, input_tensor)
        permute_dict = permute_func.compute_permute_dict()
        model = permute_func.apply_permutations(permute_dict, ignored_keys=[('conv1.weight', 'in_channels'), ('fc.weight', 'out_channels'), ('fc.bias', 'out_channels')])
        print("TV permutated model: ", compute_tv_loss_for_network(model, lambda_tv=1.0).item())
    return model    


def create_model_cifar100(model_name, hidden_dim,num_param, prior=True ,single_block=False,bottom_up=False ,path=None, smooth=False, fuse=True):
    """
    Create a model based on the specified name.

    :param model_name: String that specifies the model to use.
    :param path: Optional path for the model's weights.
    :return: The initialized model.
    """
    if model_name == "ResNet20":
        model = cifar100_resnet20(hidden_dim=hidden_dim,prior=prior, num_param=num_param, bottom_up=bottom_up, single_block=single_block,) 
    elif model_name == "ResNet32":  # Add other models as you support them
        model = cifar100_resnet32(hidden_dim=hidden_dim,prior=prior, num_param=num_param, bottom_up=bottom_up, single_block=single_block,)
    elif model_name == "ResNet44":  # Add other models as you support them
        model = cifar100_resnet44(hidden_dim=hidden_dim,prior=prior, num_param=num_param, bottom_up=bottom_up, single_block=single_block,)
    elif model_name == "ResNet56":  # Add other models as you support them
        model = cifar100_resnet56(hidden_dim=hidden_dim,prior=prior, num_param=num_param, bottom_up=bottom_up, single_block=single_block,)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    if path and not smooth and not fuse:
        if os.path.exists(path):
            #model = torch.load(path,weights_only=False)
            fuse_module(model)
            print("Loading model from", path)
            state_dict = torch.load(path, map_location=torch.device('cpu'),weights_only=True)
            load_checkpoint(model, state_dict)
            return model
        
    if path:
        print("Loading model from", path)
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        # model.load_state_dict(state_dict)
        load_checkpoint(model, state_dict)
    if fuse:
        fuse_module(model)
    if smooth:   
        print("Smooth the parameters of the model")
        input_tensor = torch.randn(1, 3, 32, 32)
        permute_func = PermutationManager(model, input_tensor)
        permute_dict = permute_func.compute_permute_dict()
        model = permute_func.apply_permutations(permute_dict, ignored_keys=[('conv1.weight', 'in_channels'), ('fc.weight', 'out_channels'), ('fc.bias', 'out_channels')])
    
   # elif path and not smooth:
   #     fuse_module(model)
   #     
   #     print("Loading model from", path)
   #     state_dict = torch.load(path, map_location=torch.device('cpu'))
   #     load_checkpoint(model, state_dict)
   # elif not path and smooth:
   #     fuse_module(model)
   #     print("Smooth the parameters of the model")
   #     input_tensor = torch.randn(1, 3, 32, 32)
   #     permute_func = PermutationManager(model, input_tensor)
   #     permute_dict = permute_func.compute_permute_dict()
   #     model = permute_func.apply_permutations(permute_dict, ignored_keys=[('conv1.weight', 'in_channels'), ('fc.weight', 'out_channels'), ('fc.bias', 'out_channels')])
        
    return model

def check_downsample_identity(model):
    for name, m in model.named_modules():
        if isinstance(m, nn.Sequential) and hasattr(m, '_skip_init'):
            print(f"Checking downsample in {name}")
            conv = m[0]
            bn = m[1]
            # Check conv weights
            is_identity = torch.allclose(
                conv.weight.data.view(64, 64), 
                torch.eye(64),
                rtol=1e-5
            )
            # Check BN parameters
            bn_identity = (
                torch.allclose(bn.weight.data, torch.ones_like(bn.weight.data)) and
                torch.allclose(bn.bias.data, torch.zeros_like(bn.bias.data)) and
                torch.allclose(bn.running_mean, torch.zeros_like(bn.running_mean)) and
                torch.allclose(bn.running_var, torch.ones_like(bn.running_var))
            )
            print(f"Conv is identity: {is_identity}")
            print(f"BN is identity: {bn_identity}")
   
def create_model_cifar100_slim(model_name, hidden_dim, num_param ,path=None, smooth=False, fuse=True):
    """
    Create a model based on the specified name.

    :param model_name: String that specifies the model to use.
    :param path: Optional path for the model's weights.
    :return: The initialized model.
    """
    #if model_name == "ResNet20":
    #    model = cifar100_resnet20(hidden_dim=hidden_dim)  # Adjust as needed based on your model's constructor
    #elif model_name == "ResNet32":  # Add other models as you support them
    #    model = cifar100_resnet32(hidden_dim=hidden_dim)
    #elif model_name == "ResNet44":  # Add other models as you support them
    #    model = cifar100_resnet44(hidden_dim=hidden_dim)
    if model_name == "ResNet56":  # Add other models as you support them
        model = cifar100_resnet56_slim(hidden_dim=hidden_dim, num_param = num_param)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    #check_downsample_identity(model)
    
    if path and not smooth and not fuse:
        if os.path.exists(path):
            #model = torch.load(path,weights_only=False)
            fuse_module(model)
            state_dict = torch.load(path, map_location=torch.device('cpu'))
            load_checkpoint(model, state_dict)
            return model
        
    if path:
        print("Loading model from", path)
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        # model.load_state_dict(state_dict)
        load_checkpoint(model, state_dict)
    if fuse:
        fuse_module(model)
    if smooth:   
        print("Smooth the parameters of the model")
        input_tensor = torch.randn(1, 3, 32, 32)
        permute_func = PermutationManager(model, input_tensor)
        permute_dict = permute_func.compute_permute_dict()
        model = permute_func.apply_permutations(permute_dict, ignored_keys=[('conv1.weight', 'in_channels'), ('fc.weight', 'out_channels'), ('fc.bias', 'out_channels')])
    
    return model
   

def create_model_imagenet(model_name, hidden_dim, path=None, smooth=None):
    """
    Create a model based on the specified name.

    :param model_name: String that specifies the model to use.
    :param path: Optional path for the model's weights.
    :return: The initialized model.
    """
    if model_name == "ResNet18":
        model = resnet18_imagenet(hidden_dim=hidden_dim, pretrained=True)  # Adjust as needed based on your model's constructor
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    fuse_module(model)

    return model

def create_model_tinyimagenet(model_name, hidden_dim, path=None, smooth=False):
    """
    Create a model based on the specified name.

    :param model_name: String that specifies the model to use.
    :param path: Optional path for the model's weights.
    :return: The initialized model.
    """
    #if model_name == "ResNet18":
    #    model = resnet18_tinyimagenet(hidden_dim)  # Adjust as needed based on your model's constructor
    if model_name == "ResNet56":
        model = tinyimagenet_resnet56(hidden_dim)  # Adjust as needed based on your model's constructor
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    if path and smooth:
        print("Loading model from", path)
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        # model.load_state_dict(state_dict)
        load_checkpoint(model, state_dict)
    
        fuse_module(model)
        
        print("Smooth the parameters of the model")
        input_tensor = torch.randn(1, 3, 64, 64)
        permute_func = PermutationManager(model, input_tensor)
        permute_dict = permute_func.compute_permute_dict()
        model = permute_func.apply_permutations(permute_dict, ignored_keys=[('conv1.weight', 'in_channels'), ('fc.weight', 'out_channels'), ('fc.bias', 'out_channels')])
    
    elif path and not smooth:
        fuse_module(model)
        
        print("Loading model from", path)
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        load_checkpoint(model, state_dict)
    elif not path and smooth:
        fuse_module(model)
        print("Smooth the parameters of the model")
        input_tensor = torch.randn(1, 3, 64, 64)
        permute_func = PermutationManager(model, input_tensor)
        permute_dict = permute_func.compute_permute_dict()
        model = permute_func.apply_permutations(permute_dict, ignored_keys=[('conv1.weight', 'in_channels'), ('fc.weight', 'out_channels'), ('fc.bias', 'out_channels')])
        

    return model