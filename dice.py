"""
Created on Wed Feb 13 2019
@author: ynomura
"""

import numpy as np
import torch
import torch.nn as nn

SMOOTH = 1.0e-6


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):

        eps = 0.0001
        intersection = torch.dot(pred.view(-1), target.view(-1))
        union = torch.sum(pred) + torch.sum(target) + eps

        return -(2 * intersection + eps) / (union + eps)


def dice_numpy(input: np.array, target: np.array):

    eps = 0.0001
    intersection = (input * target).sum()
    union = input.sum() + target.sum()

    dice_coef = (2 * intersection + eps) / (union + eps)

    return dice_coef
