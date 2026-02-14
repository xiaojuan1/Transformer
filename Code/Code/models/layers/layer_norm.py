"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn
#对最后一个维度做标准化
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))#[512]
        self.beta = nn.Parameter(torch.zeros(d_model))#[512]
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)#[2,3,4]-》[2,3,1]保留形状对最后一个维度取平均
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension. 

        out = (x - mean) / torch.sqrt(var + self.eps)#防止var=0 变成除以0
        out = self.gamma * out + self.beta#加入两个可学习参数，调整分布不是强行到0，1
        return out
