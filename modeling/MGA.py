# -*- coding: utf-8 -*-
# @Time    : 2022/5/23 15:54
# @Author  : hx
# @FileName: MGA.py
# @Software: PyCharm
import copy
import os
import torch
from torch import nn

from torchvision.models.resnet import resnet50, Bottleneck
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
import torch.nn.functional as F
import math, copy

class Attention_ViT(nn.Module):
    def __init__(self, dim, heads = 8,  dropout = 0.):
        super().__init__()
        assert dim % heads == 0
        # We assume d_v always equals d_k
        self.d_k = dim // heads
        project_out = not (heads == 1)

        self.heads = heads
        self.scale = self.d_k ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, dim*3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = [x.view(x.size(0), -1, self.heads, self.d_k).transpose(1, 2)
                             for x in qkv]
        # print('after transform query: ' + str(q.size()))  # (batch_size, seq_length, d_model)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(x.size(0), -1, self.heads * self.d_k)
        return self.to_out(out)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention',[batch,head,HW,d_k]"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads.,d_model=C"
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4) # create 4 linear layers
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, mask=None):
        key, value = query, query
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)

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

class MultiHeadAttention_nolinear(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads.,d_model=C"
        super(MultiHeadAttention_nolinear, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        #self.linears = clones(nn.Linear(d_model, d_model), 4) # create 4 linear layers
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, mask=None):
        key, value = query, query
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)

        #print ('Before transform query: ' + str(query.size())) # (batch_size, seq_length, d_model)

        #query, key, value = [l(x) for l, x in zip(self.linears, (query, key, value))] # (batch_size, seq_length, d_model), use first 3 self.linears
        query, key, value = [x.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for x in (query, key, value)] # (batch_size, h, seq_length, d_k)

        #print ('After transform query: ' + str(query.size()))

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)#(batch, channel, head*channel)
        return x # self.linears[-1](x)

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads.,d_model=C"
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4) # create 4 linear layers
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        value = key  # 来自同一之路。
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)

        # print ('Before transform query: ' + str(query.size())) # (batch_size, seq_length, d_model)

        query, key, value = [l(x) for l, x in zip(self.linears, (query, key, value))] # (batch_size, seq_length, d_model), use first 3 self.linears
        query, key, value = [x.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for x in (query, key, value)] # (batch_size, h, seq_length, d_k)

        # print ('After transform query: ' + str(query.size()))

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)#(batch, channel, head*channel)
        return self.linears[-1](x)

class TransformerBlock(nn.Module):
    r""" Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads,
                 mlp_ratio=2.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        #self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadAttention(num_heads,dim)

        #self.norm2 = norm_layer(dim)
        #mlp_hidden_dim = int(dim * mlp_ratio)
        #self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)


    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)

        # MultiHeadAttention
        x = self.attn(self.norm1(x))+x
        # FFN
        #x = self.mlp(self.norm2(x))+x

        return x.permute(0, 2, 1).view(B, C, H, W)  # 恢复张量维度
class TransformerBlock_ViT(nn.Module):
    r""" Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads,
                 mlp_ratio=2.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,dropout=0.):
        super().__init__()
        #self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = Attention_ViT(dim, num_heads,dropout=dropout)

        #self.norm2 = norm_layer(dim)
        #mlp_hidden_dim = int(dim * mlp_ratio)
        #self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)


    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)

        # MultiHeadAttention
        x = self.attn(self.norm1(x))+x
        # FFN
        #x = self.mlp(self.norm2(x))+x

        return x.permute(0, 2, 1).view(B, C, H, W)  # 恢复张量维度
class CrossTransformerBlock(nn.Module):
    r""" Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads,
                 mlp_ratio=2.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, dropout=0.):
        super().__init__()
        #self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadCrossAttention(num_heads,dim,dropout=dropout)

        #self.norm2 = norm_layer(dim)
        #mlp_hidden_dim = int(dim * mlp_ratio)
        #self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)


    def forward(self, query, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)
        query = query.view(B, C, -1).permute(0, 2, 1)

        # MultiHeadAttention
        query = self.attn(self.norm1(query), self.norm1(x))+query # 当前支路作为query
        # FFN
        #x = self.mlp(self.norm2(x))+x

        return query.permute(0, 2, 1).view(B, C, H, W)  # 恢复张量维度


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
