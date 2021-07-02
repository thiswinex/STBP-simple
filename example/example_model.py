import torch
import torch.nn as nn
import stbp as st
import torch.nn.functional as F

steps = st.get_args()['steps']

class NMNISTNet(nn.Module):  # Example net for N-MNIST
    def __init__(self):
        super(NMNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 20, 3, 1, padding=0)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(20, 50, 3, 1)
        self.pool2 = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(8 * 8 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

        self.conv1_s = st.tdLayer(self.conv1)
        self.pool1_s = st.tdLayer(self.pool1)
        self.conv2_s = st.tdLayer(self.conv2)
        self.pool2_s = st.tdLayer(self.pool2)
        self.fc1_s = st.tdLayer(self.fc1)
        self.fc2_s = st.tdLayer(self.fc2)

        self.spike = st.LIFSpike()
    
    def forward(self, x):
        x = self.conv1_s(x)
        x = self.spike(x)
        x = self.pool1_s(x)
        x = self.spike(x)
        x = self.conv2_s(x)
        x = self.spike(x)
        x = self.pool2_s(x)
        x = self.spike(x)
        x = x.view(x.shape[0], -1, x.shape[4])
        x = self.fc1_s(x)
        x = self.spike(x)
        x = self.fc2_s(x)
        x = self.spike(x)
        out = torch.sum(x, dim=2) / steps  # [N, neurons, steps]
        return out


mnist_seq_net = nn.Sequential(
    st.BroadCast(),
    st.tdLayer(nn.Conv2d(1, 15, 5, 1, 2, bias=None)),
    st.LIFSpike(),
    st.tdLayer(nn.AvgPool2d(2)),
    st.tdLayer(nn.Conv2d(15, 40, 5, 1, 2, bias=None)),
    st.LIFSpike(),
    st.tdLayer(nn.AvgPool2d(2)),
    nn.Flatten(1, 3),
    st.tdLayer(nn.Linear(7 * 7 * 40, 300)),
    st.LIFSpike(),
    st.tdLayer(nn.Linear(300, 10)),
    st.RateCoding(),
)



class MNISTNet(nn.Module):  # Example net for MNIST
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 15, 5, 1, 2, bias=None)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(15, 40, 5, 1, 2, bias=None)
        self.pool2 = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(7 * 7 * 40, 300)
        self.fc2 = nn.Linear(300, 10)

        self.conv1_s = st.tdLayer(self.conv1)
        self.pool1_s = st.tdLayer(self.pool1)
        self.conv2_s = st.tdLayer(self.conv2)
        self.pool2_s = st.tdLayer(self.pool2)
        self.fc1_s = st.tdLayer(self.fc1)
        self.fc2_s = st.tdLayer(self.fc2)

        self.spike = st.LIFSpike()
        
    def forward(self, x):
        x = self.conv1_s(x)
        x = self.spike(x)
        x = self.pool1_s(x)
        x = self.spike(x)
        x = self.conv2_s(x)
        x = self.spike(x)
        x = self.pool2_s(x)
        x = self.spike(x)
        x = x.view(x.shape[0], -1, x.shape[4])
        x = self.fc1_s(x)
        x = self.spike(x)
        x = self.fc2_s(x)
        x = self.spike(x)
        out = torch.sum(x, dim=2) / steps  # [N, neurons, steps]
        return out



class CifarNet(nn.Module):  # Example net for CIFAR10
    def __init__(self):
        super(CifarNet, self).__init__()
        #self.conv0 = nn.Conv2d(1, 1, 5, 2)
        self.conv0 = nn.Conv2d(3, 128, 3, 1, 1, bias=None)
        self.bn0 = st.tdBatchNorm(128)
        self.conv1 = nn.Conv2d(128, 256, 3, 1, 1, bias=None)
        self.bn1 = st.tdBatchNorm(256)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(256, 512, 3, 1, 1, bias=None)
        self.bn2 = st.tdBatchNorm(512)
        self.pool2 = nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(512, 1024, 3, 1, 1, bias=None)
        self.bn3 = st.tdBatchNorm(1024)
        self.conv4 = nn.Conv2d(1024, 512, 3, 1, 1, bias=None)
        self.bn4 = st.tdBatchNorm(512)
        self.fc1 = nn.Linear(8 * 8 * 512, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

        self.conv0_s = st.tdLayer(self.conv0, self.bn0)
        self.conv1_s = st.tdLayer(self.conv1, self.bn1)
        self.pool1_s = st.tdLayer(self.pool1)
        self.conv2_s = st.tdLayer(self.conv2, self.bn2)
        self.pool2_s = st.tdLayer(self.pool2)
        self.conv3_s = st.tdLayer(self.conv3, self.bn3)
        self.conv4_s = st.tdLayer(self.conv4, self.bn4)
        self.fc1_s = st.tdLayer(self.fc1)
        self.fc2_s = st.tdLayer(self.fc2)
        self.fc3_s = st.tdLayer(self.fc3)

        self.spike = st.LIFSpike()

    def forward(self, x):
        x = self.conv0_s(x)
        x = self.spike(x)
        x = self.conv1_s(x)
        x = self.spike(x)
        x = self.pool1_s(x)
        x = self.spike(x)
        x = self.conv2_s(x)
        x = self.spike(x)
        x = self.pool2_s(x)
        x = self.spike(x)
        x = self.conv3_s(x)
        x = self.spike(x)
        x = self.conv4_s(x)
        x = self.spike(x)
        x = x.view(x.shape[0], -1, x.shape[4])
        x = self.fc1_s(x)
        x = self.spike(x)
        x = self.fc2_s(x)
        x = self.spike(x)
        x = self.fc3_s(x)
        x = self.spike(x)
        out = torch.sum(x, dim=2) / steps  # [N, neurons, steps]
        return out




# ------------------- #
#   ResNet Example    #
# ------------------- #


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = st.tdBatchNorm
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, alpha=1/(2**0.5))
        self.downsample = downsample
        self.stride = stride

        self.conv1_s = st.tdLayer(self.conv1, self.bn1)
        self.conv2_s = st.tdLayer(self.conv2, self.bn2)
        self.spike = st.LIFSpike()

    def forward(self, x):
        identity = x

        out = self.conv1_s(x)
        out = self.spike(out)
        out = self.conv2_s(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.spike(out)

        return out



class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = st.tdBatchNorm
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.conv1_s = st.tdLayer(self.conv1, self.bn1)
        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.avgpool = st.tdLayer(nn.AdaptiveAvgPool2d((1, 1)))
        
        self.fc1 = nn.Linear(512 * block.expansion, 256)
        self.fc1_s = st.tdLayer(self.fc1)
        self.fc2 = nn.Linear(256, 10)
        self.fc2_s = st.tdLayer(self.fc2)
        self.spike = st.LIFSpike()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, tdBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = st.tdLayer(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion, alpha=1/(2**0.5))
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1_s(x)
        x = self.spike(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1, x.shape[4])
        x = self.fc1_s(x)
        x = self.spike(x)
        x = self.fc2_s(x)
        x = self.spike(x)
        out = torch.sum(x, dim=2) / steps
        return out

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model



def resnet19(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [3, 3, 2], pretrained, progress,
                   **kwargs)