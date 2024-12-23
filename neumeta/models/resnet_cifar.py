'''
Modified from https://raw.githubusercontent.com/pytorch/vision/v0.9.1/torchvision/models/resnet.py

BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
import sys
import torch
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
    

from .utils import load_checkpoint
from functools import partial
from typing import Dict, Type, Any, Callable, Union, List, Optional


cifar10_pretrained_weight_urls = {
    'resnet20': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet20-4118986f.pt',
    'resnet32': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet32-ef93fc4d.pt',
    'resnet44': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet44-2a3cabcb.pt',
    'resnet56': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet56-187c023a.pt',
}

cifar100_pretrained_weight_urls = {
    'resnet20': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet20-23dac2f1.pt',
    'resnet32': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet32-84213ce6.pt',
    'resnet44': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet44-ffe32858.pt',
    'resnet56': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet56-f2eff4c8.pt',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class BasicBlock_Resize(BasicBlock):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__(inplanes, planes, stride, downsample)
        self.conv2 = conv3x3(planes, inplanes)
        self.bn2 = nn.BatchNorm2d(inplanes)
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CifarResNet(nn.Module):

    def __init__(self, block, hidden_dim, num_param ,layers,single_block = False, bottom_up=False ,num_classes=10, num_layers_inr=1):
        super(CifarResNet, self).__init__()
        self.layers = layers
        self.num_param = num_param
        self.num_layers_inr = num_param# - 1
        self.single_block = single_block
        self.inplanes = 16
        self.bottom_up = bottom_up
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        
        self.set_changeable(block, hidden_dim, stride=1, num_classes=num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)  
                       
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
    def set_changeable(self, block, planes, stride, num_classes=10):
        for name, child in self.named_children():
        # Change the last block of layer3
            if name == 'layer3':
                if not self.bottom_up:
                    print(f'Replace last {self.num_layers_inr} blocks of layer3 with new blocks of hidden dim {planes}')
                    # Get all the layers except the last block
                    layers = list(child.children())[:-self.num_layers_inr]
                    #if not layers:
                    #    #TODO: handle first block of the layer, build a custom block, inplanes:32, bottleneck, outplanes: 64
                    for i in range(self.num_layers_inr):
                        layers.append(BasicBlock_Resize(64, planes, stride))
                    # layers.append(BasicBlock_Resize(64, planes, stride))
                    self._modules[name] = nn.Sequential(*layers)
                else:
                    if not self.single_block:
                        print(f'Replace first {self.num_layers_inr} blocks of layer3 with new blocks of hidden dim {planes}')
                        # Get all the layers except the last block
                        layers = []
                        layers.append(list(child.children())[0])
                        for i in range(self.num_layers_inr):
                            layers.append(BasicBlock_Resize(64, planes, stride))

                        layers.extend(list(child.children())[self.num_layers_inr+1:])
                        self._modules[name] = nn.Sequential(*layers)
                    else:
                        print(f'Replace block number {self.num_layers_inr} blocks of layer3 with new blocks of hidden dim {planes}')
                        # Get all the layers except the last block
                        layers = list(child.children())[:self.num_layers_inr]
                        
                        layers.append(BasicBlock_Resize(64, planes, stride))

                        layers.extend(list(child.children())[self.num_layers_inr+1:])
                        self._modules[name] = nn.Sequential(*layers)
    
    @property
    def learnable_parameter(self):
        #self.keys = [k for k, w in self.named_parameters() if k.startswith(f'layer3.{self.layers[-1]-1}') ]
        if not self.bottom_up:
            self.keys = [
                k for k, _ in self.named_parameters()
                if any(k.startswith(f'layer3.{self.layers[-1]-i}') for i in range(1, self.num_param + 1))
            ]
        else:
            if not self.single_block:
                self.keys = [
                    k for k, _ in self.named_parameters()
                    if any(k.startswith(f'layer3.{i}') for i in range(1, self.num_param + 1))
                ]
            else:
                self.keys = [
                    k for k, _ in self.named_parameters()
                    if k.startswith(f'layer3.{self.num_param}')
                ]
        return {k: v for k, v in self.state_dict().items() if k in self.keys}


class CifarResNet_slim(nn.Module):

    def __init__(self, block, hidden_dim, layers, num_param ,num_classes=10, num_layers_inr=0):
        super(CifarResNet_slim, self).__init__()
        self.layers = layers
        self.num_param = num_param
        self.num_layers_inr = num_param-1
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        
        self.set_changeable_slim(block, hidden_dim, stride=1, num_classes=num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        for m in self.modules():
            if hasattr(m, '_skip_init'):
                with torch.no_grad():
                    # Force identity initialization
                    conv = m[0]
                    bn = m[1]
                    
                    conv.weight.data.zero_()
                    conv.weight.data.add_(torch.eye(64).view(64, 64, 1, 1))
                    if conv.bias is not None:
                        conv.bias.data.zero_()
                        
                    bn.weight.data.fill_(1.0)
                    bn.bias.data.zero_()
                    bn.running_mean.zero_()
                    bn.running_var.fill_(1.0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
    def set_changeable_slim(self, block, planes, stride, num_classes=10):
        for name, child in self.named_children():
        # Change the last block of layer3
            if name == 'layer3':
                print(f'Replace last {self.num_layers_inr} block of layer3 with new blocks with bottleneck {planes}')
                # Get all the layers except the last block
                layers = list(child.children())[:-self.num_layers_inr]
                
                if self.num_layers_inr != 0:
                    #If there is an acutal bottleneck create the projection matrix for the residual
                    if planes!=64:
                        downsample = nn.Sequential(
                            conv1x1(64, planes * block.expansion, stride),
                            nn.BatchNorm2d(planes * block.expansion),
                            )
                    else:
                        #Otherwise initialize conv1x1 and BatchNorm2d to perform identity mapping 
                        downsample = nn.Sequential(
                            conv1x1(64, 64, stride),
                            nn.BatchNorm2d(64))
                        downsample._skip_init = True
                    
                    #shrinking block   
                    layers.append(BasicBlock(64, planes, stride,downsample=downsample))
                    
                    #bottleneck blocks
                    for i in range(self.num_layers_inr - 2):
                        layers.append(BasicBlock(planes, planes, stride))
                    
                    #Last block of the layer, back to original shape
                    if planes!=64:    
                        upsample = nn.Sequential(
                            conv1x1(planes * block.expansion, 64, stride),
                            nn.BatchNorm2d(64),
                            )
                    else:
                        upsample = nn.Sequential(
                            conv1x1(64, 64, stride),
                            nn.BatchNorm2d(64))
                        upsample._skip_init = True
                        
                    layers.append(BasicBlock(planes,64 ,stride,downsample=upsample))    
                else:
                    layers = list(child.children())
                self._modules[name] = nn.Sequential(*layers)
    
    @property
    def learnable_parameter(self):
        self.keys = [
            k for k, _ in self.named_parameters()
            if any(k.startswith(f'layer3.{self.layers[-1]-i}') for i in range(1, self.num_param + 1))
            #or k.startswith('fc')
        ]
        return {k: v for k, v in self.state_dict().items() if k in self.keys}


def _resnet(
    arch: str,
    hidden_dim: int,
    num_param:int,
    layers: List[int],
    model_urls: Dict[str, str],
    bottom_up: bool = False,
    single_block: bool = False,
    progress: bool = True,
    pretrained: bool = True,
    **kwargs: Any
) -> CifarResNet:
    model = CifarResNet(BasicBlock, hidden_dim, num_param, layers,single_block ,bottom_up,**kwargs)
    if pretrained:
        print("Loading pretrained weights for {}".format(arch))
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        # model.load_state_dict(state_dict)
        load_checkpoint(model, state_dict)
        
    return model

def _resnet_slim(
    arch: str,
    hidden_dim: int,
    layers: List[int],
    model_urls: Dict[str, str],
    progress: bool = True,
    pretrained: bool = True,
    num_param : int = 1,
    **kwargs: Any
) -> CifarResNet_slim:
    model = CifarResNet_slim(BasicBlock, hidden_dim,layers,num_param, **kwargs)
    if pretrained:
        print("Loading pretrained weights for {}".format(arch))
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        # model.load_state_dict(state_dict)
        load_checkpoint(model, state_dict)
        
    return model

# Functions for CIFAR-10
def cifar10_resnet20(hidden_dim, num_classes=10, pretrained=True, *args, **kwargs):
    return _resnet(arch="resnet20", 
                   hidden_dim=hidden_dim,
                   layers=[3]*3,  # Indicates the repetitions of certain block types
                   model_urls=cifar10_pretrained_weight_urls, 
                   num_classes=num_classes, 
                   pretrained=pretrained,
                   *args, 
                   **kwargs)

def cifar10_resnet32(hidden_dim, num_classes=10, pretrained=True, *args, **kwargs):
    return _resnet(arch="resnet32", 
                   hidden_dim=hidden_dim,
                   layers=[5]*3, 
                   model_urls=cifar10_pretrained_weight_urls, 
                   num_classes=num_classes, 
                   pretrained=pretrained,
                   *args, 
                   **kwargs)

def cifar10_resnet44(hidden_dim, num_classes=10, pretrained=True, *args, **kwargs):
    return _resnet(arch="resnet44", 
                   hidden_dim=hidden_dim,
                   layers=[7]*3, 
                   model_urls=cifar10_pretrained_weight_urls, 
                   num_classes=num_classes, 
                   pretrained=pretrained,
                   *args, 
                   **kwargs)

def cifar10_resnet56(hidden_dim, num_classes=10, pretrained=True, *args, **kwargs):
    return _resnet(arch="resnet56", 
                   hidden_dim=hidden_dim,
                   layers=[9]*3, 
                   model_urls=cifar10_pretrained_weight_urls, 
                   num_classes=num_classes, 
                   pretrained=pretrained,
                   *args, 
                   **kwargs)

# Functions for CIFAR-100
def cifar100_resnet20(hidden_dim,num_param,bottom_up,single_block=False ,num_classes=100, pretrained=True,*args, **kwargs):
    return _resnet(arch="resnet20", 
                   hidden_dim=hidden_dim,
                   num_param=num_param,
                   layers=[3]*3, 
                   bottom_up=bottom_up, 
                   single_block=single_block,
                   model_urls=cifar100_pretrained_weight_urls, 
                   num_classes=num_classes, 
                   pretrained=pretrained,
                   *args, 
                   **kwargs)

def cifar100_resnet32(hidden_dim,num_param,bottom_up,single_block=False ,num_classes=100, pretrained=True,*args, **kwargs):
    return _resnet(arch="resnet32", 
                   hidden_dim=hidden_dim,
                   num_param=num_param,
                   layers=[5]*3,
                   bottom_up=bottom_up, 
                   single_block=single_block, 
                   model_urls=cifar100_pretrained_weight_urls, 
                   num_classes=num_classes, 
                   pretrained=pretrained,
                   *args, 
                   **kwargs)

def cifar100_resnet44(hidden_dim,num_param,bottom_up,single_block=False ,num_classes=100, pretrained=True,*args, **kwargs):
    return _resnet(arch="resnet44", 
                   hidden_dim=hidden_dim,
                   num_param=num_param,
                   layers=[7]*3, 
                   bottom_up=bottom_up,
                   single_block=single_block, 
                   model_urls=cifar100_pretrained_weight_urls, 
                   num_classes=num_classes, 
                   pretrained=pretrained,
                   *args, 
                   **kwargs)

def cifar100_resnet56(hidden_dim,num_param,bottom_up,single_block=False ,num_classes=100, pretrained=True,*args, **kwargs):
    return _resnet(arch="resnet56", 
                   hidden_dim=hidden_dim,
                   num_param=num_param,
                   layers=[9]*3,
                   bottom_up=bottom_up, 
                   single_block=single_block,
                   model_urls=cifar100_pretrained_weight_urls, 
                   num_classes=num_classes, 
                   pretrained=pretrained,
                   *args, 
                   **kwargs)

def cifar100_resnet56_slim(hidden_dim,num_param ,num_classes=100, pretrained=True,*args, **kwargs):
    return _resnet_slim(arch="resnet56", 
                   hidden_dim=hidden_dim,
                   layers=[9]*3, 
                   model_urls=cifar100_pretrained_weight_urls, 
                   num_classes=num_classes, 
                   pretrained=pretrained,
                   num_param=num_param,
                   *args, 
                   **kwargs)

def tinyimagenet_resnet56(hidden_dim, num_classes=200, pretrained=True,*args, **kwargs):
    return _resnet(arch="resnet56", 
                   hidden_dim=hidden_dim,
                   layers=[9]*3, 
                   model_urls=cifar100_pretrained_weight_urls, 
                   num_classes=num_classes, 
                   pretrained=pretrained,
                   *args, 
                   **kwargs)