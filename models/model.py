import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet18
from collections import OrderedDict
from torchvision.models import resnet34, resnet50

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.dropout = nn.Dropout(0.25)  #dropout
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        # shortcut = self.shortcut(x)
        out = self.conv1(out)
        out = self.dropout(out)  # dropout
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        mlp_dim = 2 * 512 * block.expansion
        self.linear2 = nn.Sequential(
            nn.Linear(512*block.expansion , mlp_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_dim, num_classes)
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, lin=0, lout=5):
        out = x
        if lin < 1 and lout > -1:
            out = self.conv1(out)
            out = self.bn1(out)
            out = F.relu(out)
        if lin < 2 and lout > 0:
            out = self.layer1(out)
        if lin < 3 and lout > 1:
            out = self.layer2(out)
        if lin < 4 and lout > 2:
            out = self.layer3(out)
        if lin < 5 and lout > 3:
            out = self.layer4(out)
        if lout > 4:
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out1 = self.linear(out)
            out2 = self.linear2(out)
        return out1, out2


class Model_r18(nn.Module):
    def __init__(self, feature_dim=128, is_linear=False, num_classes=None):
        super(Model_r18, self).__init__()
        self.f = OrderedDict([])
        for name, module in resnet18().named_children(): 
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.update({name: module})
                
        # encoder
        self.f = nn.Sequential(self.f)
        # projection head
        self.g = nn.Sequential(
                    nn.Linear(512, 512, bias=False), 
                    nn.ReLU(inplace=True), 
                    nn.Linear(512, feature_dim, bias=True)
                )
        
        self.is_linear = is_linear
        if is_linear == True:
            self.fc = nn.Linear(512, num_classes)

    def forward(self, x, forward_fc=True, ignore_feat=False):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        projection = self.g(feature)
        if self.is_linear and forward_fc:
            logits = self.fc(feature)
            if ignore_feat == True:
                return projection, logits
            else:
                return feature, projection, logits
        else:
            if ignore_feat == True:
                return projection
            else:
                return feature, projection


class Model_r34(nn.Module):
    def __init__(self, feature_dim=128, is_linear=False, num_classes=None):
        super(Model_r34, self).__init__()
        self.f = OrderedDict([])
        for name, module in resnet34().named_children(): 
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.update({name: module})
                
        # encoder
        self.f = nn.Sequential(self.f)
        # projection head
        self.g = nn.Sequential(
                    nn.Linear(512, 512), 
                    nn.ReLU(inplace=True), 
                    nn.Linear(512, feature_dim)
                )
        
        self.is_linear = is_linear
        if is_linear == True:
            self.fc = nn.Linear(512, num_classes)

    def forward(self, x, forward_fc=True, ignore_feat=False):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        projection = self.g(feature)
        if self.is_linear and forward_fc:
            logits = self.fc(feature)
            if ignore_feat == True:
                return projection, logits
            else:
                return feature, projection, logits
        else:
            if ignore_feat == True:
                return projection
            else:
                return feature, projection

def ResNetT(num_classes=10):
    return ResNet(PreActBlock, [1, 1, 1, 1], num_classes=num_classes)

def ResNet18(num_classes=10):
    return ResNet(PreActBlock, [2,2,2,2], num_classes=num_classes)
    #return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes)
def ResNet34(num_classes=10):
    return ResNet(PreActBlock, [3,4,6,3], num_classes=num_classes)

def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes)

model_dict = {
    'resnet18': [ResNet18, 512],
    'resnet50': [ResNet50, 2048]
}


class SupCEResNet(nn.Module):
    """encoder + classifier"""

    def __init__(self, name='resnet18', num_classes=10):
        super(SupCEResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun(num_classes=num_classes)
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        out1, out2 = self.encoder(x)
        return out1,out2