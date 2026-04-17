"""
Created on Wed Feb 13 2019
@author: ynomura
"""

import numpy as np
import torch
import torch.nn as nn

SMOOTH = 1.0e-6

class DiceCELoss(nn.Module):
    def __init__(self):
        super(DiceCELoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred, target):

        eps = 0.0001
        celoss = self.cross_entropy(pred, target)
        intersection = torch.dot(pred.view(-1), target.view(-1))
        union = torch.sum(pred) + torch.sum(target) + eps
        diceloss = -(2 * intersection + eps) / (union + eps)
        
        return diceloss + celoss

