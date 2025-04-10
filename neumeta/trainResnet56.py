import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from sklearn.metrics import accuracy_score
import wandb
from torch.optim import AdamW, SGD
import time
from torchsummary import summary

from neumeta.utils.other_utils import (
    parse_args, set_seed, AverageMeter,print_omegaconf
)
from neumeta.models.resnet_cifar import cifar100_resnet56

def save_checkpoint(filepath, model, optimizer,scheduler, epoch, best_acc):
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
        'best_acc': best_acc
    }

    torch.save(checkpoint, filepath)

def train_epoch(epoch, model, train_loader, criterion, optimizer, device, args):
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        losses.update(loss.item(), data.size(0))
        
        results=torch.argmax(output,dim=1)
        train_acc=accuracy_score(results.cpu(), target.cpu())*100
        accuracies.update(train_acc)
        # Backward pass
        loss.backward()    
        optimizer.step()
            
        if batch_idx % args.experiment.log_interval == 0:
            print(f"Epoch:{epoch} Iteration {batch_idx}: Loss = {losses.avg:.4f}, Accuracy = {accuracies.avg:.2f}, Learning rate = {optimizer.param_groups[0]['lr']:.4e}") 
            if not args.experiment.debug:
                wandb.log({
                    "train loss": losses.avg,
                    "train accuracy": accuracies.avg,
                    "learning rate": optimizer.param_groups[0]['lr']
                })
            
    return losses.avg, accuracies.avg

def validate(model, val_loader, criterion, device, epoch, args):
    model.eval()
    losses = AverageMeter()
    acc = AverageMeter()
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            losses.update(loss.item())
            
            results=torch.argmax(output,dim=1)
            val_acc=accuracy_score(results.cpu(), target.cpu())*100
            acc.update(val_acc)
    print("--------------------------------------------------------------------------------------")
    print(f'Epoch:[{epoch}/{args.experiment.num_epochs}] Validation loss: {losses.avg:.4f}, '
          f'Validation Accuracy: {acc.avg:.2f}%')
    
    if not args.experiment.debug:
        wandb.log({
            "validation_loss": losses.avg,
            "validation_accuracy": acc.avg,
        })
        
    return losses.avg, acc.avg

def main():
    args = parse_args()
    print_omegaconf(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.training.save_model_path, exist_ok=True)
    
    # Set seeds for reproducibility
    set_seed(42)
    
    # Data loading and augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    train_dataset = CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    val_dataset = CIFAR100(root='./data', train=False, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.training.batch_size, 
                            shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=args.training.batch_size,
                          shuffle=False, num_workers=8)
    for dim in [256,192,128,64]:
        # Create model
        model = cifar100_resnet56(
            hidden_dim=dim,
            num_param=args.model.num_param,
            bottom_up=False,
            single_block=False,
            pretrained=False,
            config_args=args,
            prior=False
        ).to(device)
        summary(model,(3, 32, 32))
        run_name = f"hiddenDim{dim}_{args.experiment.name}-{time.strftime('%Y%m%d%H%M%S')}"
        if not args.experiment.debug:
            wandb.init(project="resnet", name=run_name, config=dict(args), group='cifar100', dir='/work/tesi_tsommariva')

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        if args.training.optimizer == 'adamw':
            optimizer = AdamW(model.parameters(), 
                                lr=args.training.learning_rate, 
                                weight_decay=args.training.weight_decay)  
            warmup_epochs = args.training.get('warmup_epochs', 5)
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_epochs)
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.training.T_max - warmup_epochs),eta_min=(args.training.learning_rate * 0.1))
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])        
            
        elif args.training.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), 
                                        lr=0.1, 
                                        momentum=args.training.get('momentum', 0.9),
                                        weight_decay=0.0005, nesterov=True)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200,eta_min=0)

        # Training loop
        best_acc = 0
        start_epoch = 0

        if args.resume_from:
            checkpoint = torch.load(args.resume_from, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']

        for epoch in range(start_epoch, args.experiment.num_epochs):
            train_loss, train_acc= train_epoch(
                epoch, model, train_loader, criterion, optimizer, device, args)

            val_loss, val_acc = validate(
                model, val_loader, criterion, device, epoch, args)
            
            scheduler.step()

            # Save checkpoint
            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(
                    os.path.join(args.training.save_model_path, f'dim{dim}_best_model.pth'),
                    model, optimizer, scheduler, epoch, best_acc
                )
            else:
                save_checkpoint(
                    os.path.join(args.training.save_model_path, f'dim{dim}_last_model.pth'),
                    model, optimizer, scheduler, epoch, best_acc
                )
            print(f"best accuracy: {best_acc:.2f}")
            print("--------------------------------------------------------------------------------------")
        if not args.experiment.debug:
            wandb.finish()   

if __name__ == "__main__":
    main()
