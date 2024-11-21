import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import argparse
from collections import OrderedDict

# Define the LambdaLayer
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

# Define the BasicBlock with Option A for the shortcut
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes,
            kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes,
            kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                # For CIFAR10 ResNet paper uses option A.
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant", 0
                    )
                )
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes, planes * self.expansion,
                        kernel_size=1, stride=stride,
                        bias=False
                    ),
                    nn.BatchNorm2d(planes * self.expansion)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Define the ResNet model matching the training implementation
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3,
            stride=1, padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(
            block, 16, num_blocks[0], stride=1
        )
        self.layer2 = self._make_layer(
            block, 32, num_blocks[1], stride=2
        )
        self.layer3 = self._make_layer(
            block, 64, num_blocks[2], stride=2
        )
        self.linear = nn.Linear(64, num_classes)

        # Initialize weights (optional, not needed if loading weights)
        # self.apply(self._weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for current_stride in strides:
            layers.append(block(self.in_planes, planes, current_stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    # Optional weights initialization method
    # def _weights_init(self, m):
    #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #         nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # Global average pooling
        out = F.avg_pool2d(out, out.size(3))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])

def main():
    parser = argparse.ArgumentParser(description='Evaluate ResNet20 on CIFAR-10')
    parser.add_argument('--weights', default='resnet20_weights.pth', type=str,
                        help='Path to the ResNet20 weights file')
    args = parser.parse_args()

    # Device configuration (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Initialize and load the model
    model = resnet20().to(device)

    # Load the checkpoint
    checkpoint = torch.load(args.weights, map_location=device)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Handle 'module.' prefix if present in the state_dict keys
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[len('module.'):]  # remove 'module.' prefix
        else:
            name = k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    # Define the CIFAR-10 test dataset and loader
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),  # mean
            (0.2023, 0.1994, 0.2010)   # std
        ),
    ])

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False,
        download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100,
        shuffle=False, num_workers=2
    )

    # Evaluate the model on the test dataset
    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    # Print the accuracy
    accuracy = 100 * correct / total
    print('Accuracy on the CIFAR-10 test set: {:.2f}%'.format(accuracy))

if __name__ == '__main__':
    main()
