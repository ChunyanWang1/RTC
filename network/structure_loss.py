import scipy
import torch 
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from scipy import special

eps = 1e-12

def CalPairwise(dist):
    dist[dist < 0] = 0
    Pij = torch.exp(-dist)
    return Pij

def Distance_squared(x, y, featdim=1):
    b,c,h,w=x.size()
    # x=x.view(b,c,h*w)
    # y=y.view(b,c,h*w)
    # x=torch.mean(x,dim=-1)
    # y=torch.mean(y,dim=-1)
    x = torch.flatten(x, start_dim=1)
    y= torch.flatten(y, start_dim=1)

    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    d = dist.clamp(min=eps)
    d[torch.eye(d.shape[0]) == 1] = eps
    return d


def loss_structure(feat1, feat2):#,cam1,cam2
    
    # q1 = CalPairwise(Distance_squared(feat1, feat1))
    # q2 = CalPairwise(Distance_squared(feat2, feat2))
    q1 = CalPairwise(Distance_squared(feat1, feat1))
    q2 = CalPairwise(Distance_squared(feat2, feat2))
    return -1 * (q1 * torch.log(q2 + eps)).mean()

def loss_structure1(feat1, feat2):
    q1 = CalPairwise(Distance_squared(feat1, feat1))
    q2 = CalPairwise(Distance_squared(feat2, feat2))
    return -1 * (q1 * torch.log(q2 + eps)).mean()

