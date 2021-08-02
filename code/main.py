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

import argparse

from cal import test

parser = argparse.ArgumentParser(
    description="Pytorch Detecting Out-of-distribution examples in neural networks"
)

parser.add_argument(
    "--nn", default="densenet10", type=str, help="neural network name and training set"
)
parser.add_argument(
    "--out_dataset", default="Imagenet", type=str, help="out-of-distribution dataset"
)
parser.add_argument(
    "--magnitude", default=0.0014, type=float, help="perturbation magnitude"
)
parser.add_argument("--temperature", default=1000, type=int, help="temperature scaling")
parser.add_argument("--gpu", default=0, type=int, help="gpu index")
parser.set_defaults(argument=True)


# Setting the name of neural networks

# Densenet trained on CIFAR-10:         densenet10
# Densenet trained on CIFAR-100:        densenet100
# Wide-ResNet trained on CIFAR-10:    wideresnet10
# Wide-ResNet trained on CIFAR-100:   wideresnet100
# nnName = "densenet10"

# Setting the name of the out-of-distribution dataset

# Tiny-ImageNet (crop):     Imagenet
# Tiny-ImageNet (resize):   Imagenet_resize
# LSUN (crop):              LSUN
# LSUN (resize):            LSUN_resize
# iSUN:                     iSUN
# Gaussian noise:           Gaussian
# Uniform  noise:           Uniform
# dataName = "Imagenet"


# Setting the perturbation magnitude
# epsilon = 0.0014

# Setting the temperature
# temperature = 1000
def main():
    global args
    args = parser.parse_args()
    test(args.nn, args.out_dataset, args.gpu, args.magnitude, args.temperature)


if __name__ == "__main__":
    main()
