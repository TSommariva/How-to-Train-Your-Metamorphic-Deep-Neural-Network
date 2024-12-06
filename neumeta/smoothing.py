from neumeta.models import create_model_cifar10
import torch
import torch.nn as nn
from smooth.permute import PermutationManager, compute_tv_loss_for_network
from neumeta.utils import parse_args, print_omegaconf, set_seed, get_cifar10,validate_single


device = "cpu"

# Parse the arguments
args = parse_args()
# Print the arguments
print_omegaconf(args)
# Set the random seed
set_seed(args.experiment.seed)


# Create the model for CIFAR10
model = create_model_cifar10(args.model.type, 
        hidden_dim=args.dimensions.start, 
        path=args.model.pretrained_path, 
        smooth=False,
        fuse=True).to(device)

model.eval()  # Set to evaluation mode

train_loader, val_loader = get_cifar10(args.training.batch_size, 
                                           strong_transform=args.training.get('strong_aug', None))

val_loss, val_acc = validate_single(model, val_loader, nn.CrossEntropyLoss(), args=args, device=device)
print(f"Pretrained model Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc*100:.2f}%")

print("Smooth the parameters of the model")
print("TV original model: ", compute_tv_loss_for_network(model, lambda_tv=1.0).item())
input_tensor = torch.randn(1, 3, 32, 32)
permute_func = PermutationManager(model, input_tensor)
permute_dict = permute_func.compute_permute_dict()
model = permute_func.apply_permutations(permute_dict, ignored_keys=[('conv1.weight', 'in_channels'), ('fc.weight', 'out_channels'), ('fc.bias', 'out_channels')])
print("TV permutated model: ", compute_tv_loss_for_network(model, lambda_tv=1.0).item())

val_loss, val_acc = validate_single(model, val_loader, nn.CrossEntropyLoss(), args=args, device=device)
print(f"Smoothed model Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc*100:.2f}%")


torch.save(model,"neumeta/pretrained_models/cifar10_resnet20-smoothed.pt")

model = create_model_cifar10(args.model.type, 
        hidden_dim=args.dimensions.start, 
        path="neumeta/pretrained_models/cifar10_resnet20-smoothed.pt", 
        smooth=False, fuse=False).to(device)

val_loss, val_acc = validate_single(model, val_loader, nn.CrossEntropyLoss(), args=args, device=device)
print(f"Saved model Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc*100:.2f}%")