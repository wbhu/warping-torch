#!/usr/bin/env python
"""
    File Name   :   warping-torch-backward_warp
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

import torch.nn as nn
from torch.nn import functional as F
import torch
import numpy as np


class BackwardWarp(nn.Module):
    def __init__(self, height=256, width=256, cuda=True):
        super(BackwardWarp, self).__init__()
        self.H = height
        self.W = width
        remapW, remapH = np.meshgrid(np.arange(width), np.arange(height))
        reGrid = np.stack((2.0 * remapW / max(width - 1, 1) - 1.0, 2.0 * remapH / max(height - 1, 1) - 1.0), axis=-1)
        reGrid = reGrid[np.newaxis, ...]
        self.grid = torch.from_numpy(reGrid.astype(np.float32))
        self.cuda = cuda

    def forward(self, x, flow):
        # x is img, in N*C*H*W format
        # flow is in N*2*H*W format
        # flow[:,0,:,:] is the W direction (X axis) flow map !!
        flow_tmp = flow.clone()
        flow_tmp[:, 0, :, :] /= self.W
        flow_tmp[:, 1, :, :] /= self.H
        if self.cuda:
            grid = self.grid.cuda(flow_tmp.get_device()) + 2.0 * flow_tmp.permute(0, 2, 3, 1)
        else:
            grid = self.grid + 2.0 * flow_tmp.permute(0, 2, 3, 1)
        return F.grid_sample(x, grid, padding_mode='zeros', mode='bilinear', align_corners=True)
