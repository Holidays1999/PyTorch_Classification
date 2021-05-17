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
from models.resnet import ResNet18, ResNet50, ResNet101, WideResNet50_2, WideResNet101_2
from models.resnet_cifar import cResNet18, cResNet50, cResNet101
from models.vgg import vgg19, vgg19_bn, vgg16, vgg16_bn, vgg13, vgg13_bn, vgg11, vgg11_bn
from models.densenet import DenseNet121, DenseNet161, DenseNet169, DenseNet201


__all__ = [
    "ResNet18",
    "ResNet50",
    "ResNet101",
    "cResNet18",
    "cResNet50",
    "WideResNet50_2",
    "WideResNet101_2",
    "vgg19",
    "vgg19_bn",
    "vgg16",
    "vgg16_bn",
    "vgg13",
    "vgg13_bn",
    "vgg11",
    "vgg11_bn",
    "DenseNet121",
    "DenseNet161",
    "DenseNet169",
    "DenseNet201",
]
