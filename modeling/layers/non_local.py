# encoding: utf-8

import torch
from torch import nn


class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = 1

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
                :param x: (b, c, h, w)
                :return x: (b, c, h, w)
        '''
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # (b, c/2, h*w)
        g_x = g_x.permute(0, 2, 1)  # (b, h*w, c/2)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)  # (b, c/2, h*w)
        theta_x = theta_x.permute(0, 2, 1)  # (b, h*w, c/2)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  # (b, c/2, h*w)

        f = torch.matmul(theta_x, phi_x) # (b, h*w, h*w)
        N = f.size(-1) # h*w
        f_div_C = f / N # (b, h*w, h*w)

        y = torch.matmul(f_div_C, g_x)  # (b, h*w, c/2)
        y = y.permute(0, 2, 1).contiguous()  # (b, c/2, h*w)
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])  # (b, c/2, h, w)
        W_y = self.W(y)  # (b, c, h, w)
        z = W_y + x  # (b, c, h, w)
        return z


import torch.nn.functional as F
import math, copy

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1) # query=(batch, head, channel, dimension)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) #(batch, head, channel, channel)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)#(batch, head, channel, channel)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4) # create 4 linear layers
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)
        key, value = query, query
        print ('Before transform query: ' + str(query.size())) # (batch_size, seq_length, d_model)

        query, key, value = [l(x) for l, x in zip(self.linears, (query, key, value))] # (batch_size, seq_length, d_model), use first 3 self.linears
        query, key, value = [x.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for x in (query, key, value)] # (batch_size, h, seq_length, d_k)

        print ('After transform query: ' + str(query.size()))

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)#(batch, channel, head*channel)
        return self.linears[-1](x)
