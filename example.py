#!/usr/bin/env python
"""
    File Name   :   warping-torch-example
    date        :   4/7/2020
    Author      :   wenbo
    Email       :   huwenbodut@gmail.com
    Description :
                              _     _
                             ( |---/ )
                              ) . . (
________________________,--._(___Y___)_,--._______________________
                        `--'           `--'
"""
import os

import torch
import torch.nn as nn
from torch import optim
import cv2
import numpy as np
import torch.nn.functional as F
import pdb
from forward_warp import ForwardWarpStereo


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def img2tensor(img, cuda=True):
    img_t = np.expand_dims(img.transpose(2, 0, 1), axis=0)
    img_t = torch.from_numpy(img_t.astype(np.float32))
    if cuda:
        img_t = img_t.cuda(non_blocking=True)
    return img_t


def tensor2img(img_t):
    if len(img_t.shape) == 4:
        img = img_t[0].detach().cpu().numpy()
    elif len(img_t.shape) == 3:
        img = img_t.detach().cpu().numpy()
    else:
        raise NotImplementedError
    img = img.transpose(1, 2, 0)
    return img


# pdb.set_trace()
l = cv2.imread('test_data/0108_L.png') / 255.
r = cv2.imread('test_data/0108_R.png') / 255.
l_tensor = img2tensor(l)
r_tensor = img2tensor(r)
disp = np.load('test_data/0108_L_disp.npy')
disp = torch.from_numpy(disp)

if torch.cuda.is_available():
    l_tensor = l_tensor.cuda()
    r_tensor = r_tensor.cuda()
    disp = disp.cuda()

#  Test forward warping
stereo_warpper = ForwardWarpStereo(occlu_map=True)

for i in range(10):
    t = i / 10.
    l2r, occlu = stereo_warpper(l_tensor, -disp * t)
    l2r = tensor2img(l2r)
    ensure_dir('result')
    cv2.imwrite('result/l2r%2d.png' % i, l2r * 255.)
    occlu = tensor2img(occlu)
    cv2.imwrite('result/occlu%2d.png' % i, occlu * 255.)
