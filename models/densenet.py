# Copyright 2021 Zhejiang University of Techonology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import torch
import torch.nn as nn
from collections import OrderedDict
from utils.builder import get_builder
from args import args


class _DenseLayer(nn.Sequential):
    def __init__(self, builder, in_channels, growth_rate, bn_size):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', builder.batchnorm(in_channels))
        self.add_module('relu1', builder.activation())
        self.add_module('conv1', builder.conv1x1(in_channels, bn_size * growth_rate))
        self.add_module('norm2', builder.batchnorm(bn_size*growth_rate))
        self.add_module('relu2', builder.activation())
        self.add_module('conv2', builder.conv3x3(bn_size*growth_rate, growth_rate))

    # 重载forward函数
    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, builder, num_layers, in_channels, bn_size, growth_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            self.add_module('denselayer%d' % (i+1),
                            _DenseLayer(builder, in_channels+growth_rate*i,
                                        growth_rate, bn_size))


class _Transition(nn.Sequential):
    def __init__(self, builder, in_channels, out_channels):
        super(_Transition, self).__init__()
        self.add_module('norm', builder.batchnorm(in_channels))
        self.add_module('relu', builder.activation())
        self.add_module('conv', builder.conv1x1(in_channels, out_channels))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(self, builder, growth_rate=12, block_config=(6,12,24,16),
                 bn_size=4, theta=0.5):
        super(DenseNet, self).__init__()

        # 初始的卷积为filter:2倍的growth_rate
        num_init_feature = 2 * growth_rate

        self.features = nn.Sequential(OrderedDict([
            ('conv0', builder.conv7x7(3, num_init_feature)),
            ('norm0', builder.batchnorm(num_init_feature)),
            ('relu0', builder.activation()),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
             ]))

        num_feature = num_init_feature
        for i, num_layers in enumerate(block_config):
            self.features.add_module('denseblock%d' % (i+1),
                                     _DenseBlock(builder, num_layers, num_feature,
                                                 bn_size, growth_rate))
            num_feature = num_feature + growth_rate * num_layers
            if i != len(block_config)-1:
                self.features.add_module('transition%d' % (i + 1),
                                         _Transition(builder, num_feature,
                                                     int(num_feature * theta)))
                num_feature = int(num_feature * theta)

        self.features.add_module('norm5', builder.batchnorm(num_feature))
        self.features.add_module('relu5', builder.activation())
        self.features.add_module('avg_pool', nn.AdaptiveAvgPool2d((1, 1)))

        self.classifier = builder.conv1x1(num_feature, args.num_classes)

    def forward(self, x):
        features = self.features(x)
        out = self.classifier(features).view(features.size(0), -1)
        return out


# DenseNet_BC for ImageNet
def DenseNet121():
    return DenseNet(get_builder(), growth_rate=32, block_config=(6, 12, 24, 16))

def DenseNet169():
    return DenseNet(get_builder(), growth_rate=32, block_config=(6, 12, 32, 32))

def DenseNet201():
    return DenseNet(get_builder(), growth_rate=32, block_config=(6, 12, 48, 32))

def DenseNet161():
    return DenseNet(get_builder(), growth_rate=48, block_config=(6, 12, 36, 24))

# DenseNet_BC for cifar
def densenet_BC_100():
    return DenseNet(get_builder(), growth_rate=12, block_config=(16, 16, 16))