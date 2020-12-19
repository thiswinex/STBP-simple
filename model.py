import torch
import torch.nn as nn
from layers import *
import torch.nn.functional as F

class NMNISTNet(nn.Module):  # Example net for N-MNIST
    def __init__(self):
        super(NMNISTNet, self).__init__()
        self.conv1 = SpikeConv2d(2, 20, 3, 1, padding=0)
        self.pool1 = SpikeAvgPool(2)
        self.conv2 = SpikeConv2d(20, 50, 3, 1)
        self.pool2 = SpikeAvgPool(2)
        self.fc1 = SpikeLinear(8 * 8 * 50, 500)
        self.fc2 = SpikeLinear(500, 10)
        
    def initSpikeParam(self, x):

        x = self.conv1.initSpikeParam(x)
        x = self.pool1.initSpikeParam(x)
        x = self.conv2.initSpikeParam(x)
        x = self.pool2.initSpikeParam(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1.initSpikeParam(x)
        x = self.fc2.initSpikeParam(x)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        #print(x.sum())
        #print(x.shape)
        x = x.view(x.shape[0], -1, x.shape[4])
        x = self.fc1(x)
        x = self.fc2(x)
        out = torch.sum(x, dim=2) / steps  # [N, neurons, steps]
        #print(out)
        return out



class MNISTNet(nn.Module):  # Example net for MNIST
    def __init__(self):
        super(MNISTNet, self).__init__()
        #self.conv0 = nn.Conv2d(1, 1, 5, 2)
        self.conv1 = nn.Conv2d(1, 15, 5, 1, 2, bias=None)
        self.conv1_s = SpikeLayer(self.conv1)
        self.pool1 = nn.AvgPool2d(2)
        self.pool1_s = SpikeLayer(self.pool1)
        self.conv2 = nn.Conv2d(15, 40, 5, 1, 2, bias=None)
        self.conv2_s = SpikeLayer(self.conv2)
        self.pool2 = nn.AvgPool2d(2)
        self.pool2_s = SpikeLayer(self.pool2)
        self.fc1 = nn.Linear(7 * 7 * 40, 300)
        self.fc1_s = SpikeLayer(self.fc1)
        self.fc2 = nn.Linear(300, 10)
        self.fc2_s = SpikeLayer(self.fc2)
        
    def forward(self, x):
        x = self.conv1_s(x)
        x = self.pool1_s(x)
        x = self.conv2_s(x)
        x = self.pool2_s(x)
        x = x.view(x.shape[0], -1, x.shape[4])
        x = self.fc1_s(x)
        x = self.fc2_s(x)
        
        #x = x.reshape((-1,) + x.shape[1:] +(steps,))
        out = torch.sum(x, dim=2) / steps  # [N, neurons, steps]
        #print(out)
        return out



class CifarNet(nn.Module):  # Example net for CIFAR10
    def __init__(self):
        super(CifarNet, self).__init__()
        #self.conv0 = nn.Conv2d(1, 1, 5, 2)
        self.conv0 = nn.Conv2d(3, 128, 3, 1, 1, bias=None)
        self.bn0 = tdBatchNorm(128)
        self.conv1 = nn.Conv2d(128, 256, 3, 1, 1, bias=None)
        self.bn1 = tdBatchNorm(256)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(256, 512, 3, 1, 1, bias=None)
        self.bn2 = tdBatchNorm(512)
        self.pool2 = nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(512, 1024, 3, 1, 1, bias=None)
        self.bn3 = tdBatchNorm(1024)
        self.conv4 = nn.Conv2d(1024, 512, 3, 1, 1, bias=None)
        self.bn4 = tdBatchNorm(512)
        self.fc1 = nn.Linear(8 * 8 * 512, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

        self.conv0_s = SpikeLayer(self.conv0, self.bn0)
        #self.conv0_s = SpikeLayer(self.conv0)
        #self.conv1_s = SpikeLayer(self.conv1, self.bn1)
        self.conv1_s = SpikeLayer(self.conv1)
        self.pool1_s = SpikeLayer(self.pool1)
        #self.conv2_s = SpikeLayer(self.conv2, self.bn2)
        self.conv2_s = SpikeLayer(self.conv2)
        self.pool2_s = SpikeLayer(self.pool2)
        #self.conv3_s = SpikeLayer(self.conv3, self.bn3)
        self.conv3_s = SpikeLayer(self.conv3)
        #self.conv4_s = SpikeLayer(self.conv4, self.bn4)
        self.conv4_s = SpikeLayer(self.conv4)
        self.fc1_s = SpikeLayer(self.fc1)
        self.fc2_s = SpikeLayer(self.fc2)
        self.fc3_s = SpikeLayer(self.fc3)

        
    
    def forward(self, x):
        x = self.conv0_s(x)
        x = self.conv1_s(x)
        x = self.pool1_s(x)
        x = self.conv2_s(x)
        x = self.pool2_s(x)
        x = self.conv3_s(x)
        x = self.conv4_s(x)
        #print(x.sum())
        #print(x.shape)
        x = x.view(x.shape[0], -1, x.shape[4])
        x = self.fc1_s(x)
        #x = F.dropout(x, 0.5)
        x = self.fc2_s(x)
        #x = F.dropout(x, 0.5)
        x = self.fc3_s(x)
        out = torch.sum(x, dim=2) / steps  # [N, neurons, steps]
        #print(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
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




class ResNet19(nn.Module):  # Example net for CIFAR10
    def __init__(self):
        super(ResNet19, self).__init__()
        #self.conv0 = nn.Conv2d(1, 1, 5, 2)
        self.conv0 = nn.Conv2d(3, 128, 3, 1, 1, bias=None)
        self.bn0 = tdBatchNorm(128)
        self.conv1 = nn.Conv2d(128, 256, 3, 1, 1, bias=None)
        self.bn1 = tdBatchNorm(256)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(256, 512, 3, 1, 1, bias=None)
        self.bn2 = tdBatchNorm(512)
        self.pool2 = nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(512, 1024, 3, 1, 1, bias=None)
        self.bn3 = tdBatchNorm(1024)
        self.conv4 = nn.Conv2d(1024, 512, 3, 1, 1, bias=None)
        self.bn4 = tdBatchNorm(512)
        self.fc1 = nn.Linear(8 * 8 * 512, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

        self.conv0_s = SpikeLayer(self.conv0, self.bn0)
        #self.conv0_s = SpikeLayer(self.conv0)
        #self.conv1_s = SpikeLayer(self.conv1, self.bn1)
        self.conv1_s = SpikeLayer(self.conv1)
        self.pool1_s = SpikeLayer(self.pool1)
        #self.conv2_s = SpikeLayer(self.conv2, self.bn2)
        self.conv2_s = SpikeLayer(self.conv2)
        self.pool2_s = SpikeLayer(self.pool2)
        #self.conv3_s = SpikeLayer(self.conv3, self.bn3)
        self.conv3_s = SpikeLayer(self.conv3)
        #self.conv4_s = SpikeLayer(self.conv4, self.bn4)
        self.conv4_s = SpikeLayer(self.conv4)
        self.fc1_s = SpikeLayer(self.fc1)
        self.fc2_s = SpikeLayer(self.fc2)
        self.fc3_s = SpikeLayer(self.fc3)

        
    
    def forward(self, x):
        x = self.conv0_s(x)
        x = self.conv1_s(x)
        x = self.pool1_s(x)
        x = self.conv2_s(x)
        x = self.pool2_s(x)
        x = self.conv3_s(x)
        x = self.conv4_s(x)
        #print(x.sum())
        #print(x.shape)
        x = x.view(x.shape[0], -1, x.shape[4])
        x = self.fc1_s(x)
        #x = F.dropout(x, 0.5)
        x = self.fc2_s(x)
        #x = F.dropout(x, 0.5)
        x = self.fc3_s(x)
        out = torch.sum(x, dim=2) / steps  # [N, neurons, steps]
        #print(out)
        return out
