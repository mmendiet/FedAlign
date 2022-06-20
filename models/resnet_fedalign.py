'''
ResNet Construction Code
Code credit to https://github.com/FedML-AI/FedML,
https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py,
https://github.com/taoyang1122/GradAug
'''

import torch
import torch.nn as nn
from models.slimmable_ops import USBatchNorm2d, USConv2d, USLinear, make_divisible

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, width_max=1.0):
    """3x3 convolution with padding"""
    return USConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation, width_max=width_max)


def conv1x1(in_planes, out_planes, stride=1, width_max=1.0):
    """1x1 convolution"""
    return USConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, width_max=width_max)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, max_width=1.0):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = USBatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, width_max=max_width)
        self.bn1 = norm_layer(planes, width_max=max_width)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, width_max=max_width)
        self.bn2 = norm_layer(planes, width_max=max_width)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, max_width=1.0):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = USBatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, width_max=max_width)
        self.bn1 = norm_layer(width, width_max=max_width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, width_max=max_width)
        self.bn2 = norm_layer(width, width_max=max_width)
        self.conv3 = conv1x1(width, planes * self.expansion, width_max=max_width)
        self.bn3 = norm_layer(planes * self.expansion, width_max=max_width)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, KD=False, max_width=1.0):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = USBatchNorm2d
        self._norm_layer = norm_layer
        self.max_width = max_width
        self.inplanes = 16
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
        self.conv1 = USConv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False, us=[False, True], width_max=self.max_width)
        self.bn1 = USBatchNorm2d(self.inplanes, width_max=self.max_width)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d()
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = USLinear(64 * block.expansion, num_classes, us=[True, False], width_max=self.max_width)
        self.KD = KD
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) and m.affine:
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
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, width_max=self.max_width),
                norm_layer(planes * block.expansion, width_max=self.max_width),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, max_width=self.max_width))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, max_width=self.max_width))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # B x 16 x 32 x 32
        x = self.layer1(x)  # B x 16 x 32 x 32
        x = self.layer2(x)  # B x 32 x 16 x 16
        x = self.layer3(x)  # B x 64 x 8 x 8

        x = self.avgpool(x)  # B x 64 x 1 x 1
        x_f = x.view(x.size(0), -1)  # B x 64
        x = self.fc(x_f)  # B x num_classes
        if self.KD == True:
            return x_f, x
        else:
            return x

    def extract_feature(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # B x 16 x 32 x 32
        x = self.layer1(x)  # B x 16 x 32 x 32
        x2 = self.layer2(x)  # B x 32 x 16 x 16
        x3 = self.layer3(x2)  # B x 64 x 8 x 8

        x = self.avgpool(x3)  # B x 64 x 1 x 1
        x_f = x.view(x.size(0), -1)  # B x 64
        x = self.fc(x_f)  # B x num_classes
        if self.KD == True:
            return x_f, x
        else:
            return [x2, x3], x
            
    def reuse_feature(self, x, ):
        x2 = x[:, :make_divisible(x.shape[1]*self.width_mult)]
        x3 = self.layer3(x2)
        return [x2, x3]

class ImageNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, KD=False, max_width=1.0):
        super(ImageNet, self).__init__()
        if norm_layer is None:
            norm_layer = USBatchNorm2d
        self._norm_layer = norm_layer
        self.max_width = max_width
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
        self.conv1 = USConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False, us=[False, True], width_max=self.max_width)
        # self.conv1 = USConv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                            #    bias=False, us=[False, True], width_max=self.max_width)
        self.bn1 = USBatchNorm2d(self.inplanes, width_max=self.max_width)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = USLinear(512 * block.expansion, num_classes, us=[True, False], width_max=self.max_width)
        self.KD = KD
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) and m.affine:
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
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, width_max=self.max_width),
                norm_layer(planes * block.expansion, width_max=self.max_width),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, max_width=self.max_width))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, max_width=self.max_width))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # B x 16 x 32 x 32
        x = self.maxpool(x)

        x = self.layer1(x)  # B x 16 x 32 x 32
        x = self.layer2(x)  # B x 32 x 16 x 16
        x = self.layer3(x)  # B x 64 x 8 x 8
        x = self.layer4(x)

        x = self.avgpool(x)  # B x 64 x 1 x 1
        x_f = x.view(x.size(0), -1)  # B x 64
        x = self.fc(x_f)  # B x num_classes
        if self.KD == True:
            return x_f, x
        else:
            return x

    def extract_feature(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # B x 16 x 32 x 32
        x = self.maxpool(x)

        x = self.layer1(x)  # B x 16 x 32 x 32
        x = self.layer2(x)  # B x 32 x 16 x 16
        x3 = self.layer3(x)  # B x 64 x 8 x 8
        x4 = self.layer4(x3)

        x = self.avgpool(x4)  # B x 64 x 1 x 1
        x_f = x.view(x.size(0), -1)  # B x 64
        x = self.fc(x_f)  # B x num_classes
        if self.KD == True:
            return x_f, x
        else:
            return [x3, x4], x

    def reuse_feature(self, x):
        x3 = x[:, :make_divisible(x.shape[1]*self.width_mult)]
        x4 = self.layer4(x3)
        return [x3, x4]

def resnet56(class_num, pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-56 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    model = ResNet(Bottleneck, [6, 6, 6], class_num, **kwargs)
    if pretrained:
        checkpoint = torch.load(path)
        state_dict = checkpoint['state_dict']

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # name = k[7:]  # remove 'module.' of dataparallel
            name = k.replace("module.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    return model

def resnet18(class_num, pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    model = ImageNet(BasicBlock, [2, 2, 2, 2], class_num, **kwargs)
    if pretrained:
        checkpoint = torch.load(path)
        state_dict = checkpoint['state_dict']

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # name = k[7:]  # remove 'module.' of dataparallel
            name = k.replace("module.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    return model