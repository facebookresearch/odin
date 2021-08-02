# -*- coding: utf-8 -*-
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Created on Sat Sep 19 20:55:56 2015

@author: liangshiyu
"""

from __future__ import print_function

import time

import torch
from torch.nn import CrossEntropyLoss
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torchvision.transforms import Compose, Normalize, ToTensor

import calData as d
import calMetric as m

# CUDA_DEVICE = 0

start = time.time()
# loading data sets

transform = Compose(
    [
        ToTensor(),
        Normalize(
            (125.3 / 255, 123.0 / 255, 113.9 / 255),
            (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0),
        ),
    ]
)


# loading neural network

# Name of neural networks
# Densenet trained on CIFAR-10:         densenet10
# Densenet trained on CIFAR-100:        densenet100
# Densenet trained on WideResNet-10:    wideresnet10
# Densenet trained on WideResNet-100:   wideresnet100
# nnName = "densenet10"

# imName = "Imagenet"


criterion = CrossEntropyLoss()


def test(nnName, dataName, CUDA_DEVICE, epsilon, temperature):
    net1 = torch.load(
        "../models/{}.pth".format(nnName), map_location=torch.device("cpu")
    )
    # optimizer1 = optim.SGD(net1.parameters(), lr=0, momentum=0)
    # net1.cuda(CUDA_DEVICE)

    if dataName != "Uniform" and dataName != "Gaussian":
        testsetout = ImageFolder("../data/{}".format(dataName), transform=transform)
        testloaderOut = torch.utils.data.DataLoader(
            testsetout, batch_size=1, shuffle=False, num_workers=2
        )

    if nnName == "densenet10" or nnName == "wideresnet10":
        testset = CIFAR10(
            root="../data", train=False, download=True, transform=transform
        )
        testloaderIn = torch.utils.data.DataLoader(
            testset, batch_size=1, shuffle=False, num_workers=2
        )
    if nnName == "densenet100" or nnName == "wideresnet100":
        testset = CIFAR100(
            root="../data", train=False, download=True, transform=transform
        )
        testloaderIn = torch.utils.data.DataLoader(
            testset, batch_size=1, shuffle=False, num_workers=2
        )

    if dataName == "Gaussian":
        d.testGaussian(
            net1,
            criterion,
            CUDA_DEVICE,
            testloaderIn,
            testloaderIn,
            nnName,
            dataName,
            epsilon,
            temperature,
        )
        m.metric(nnName, dataName)

    elif dataName == "Uniform":
        d.testUni(
            net1,
            criterion,
            CUDA_DEVICE,
            testloaderIn,
            testloaderIn,
            nnName,
            dataName,
            epsilon,
            temperature,
        )
        m.metric(nnName, dataName)
    else:
        d.testData(
            net1,
            criterion,
            CUDA_DEVICE,
            testloaderIn,
            testloaderOut,
            nnName,
            dataName,
            epsilon,
            temperature,
        )
        m.metric(nnName, dataName)
