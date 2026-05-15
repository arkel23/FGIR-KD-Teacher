from functools import partial

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from .drop_path import DropPath


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


class AllYouNeedAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""
    def __init__(self, dim, num_heads, dropout, proj_dim=None):
        super().__init__()
        if not proj_dim:
            proj_dim = dim

        self.proj_q = nn.Linear(dim, proj_dim)
        self.proj_k = nn.Linear(dim, proj_dim)
        self.proj_v = nn.Linear(dim, proj_dim)
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads

    def forward(self, x):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)

        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])

        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))

        # this is what's used to visualize attention
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)

        return h


class PositionWiseFeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(F.gelu(self.fc1(x)))


class BlockVanilla(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, ff_dim, hidden_dropout_prob,
                 attention_probs_dropout_prob, layer_norm_eps, sd=0):
        super().__init__()
        self.attn = AllYouNeedAttention(dim, num_heads, attention_probs_dropout_prob)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)

        self.pwff = PositionWiseFeedForward(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)

        if sd > 0:
            self.drop = DropPath(sd)
        else:
            self.drop = nn.Dropout(hidden_dropout_prob)

    def forward(self, x):

        h = self.attn(self.norm1(x))
        h = self.drop(self.proj(h))
        x = x + h

        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h

        return x


class Transformer(nn.Module):
    """Transformer with Self-Attentive Blocks"""

    def __init__(self, num_layers, dim, num_heads, ff_dim, hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.0, layer_norm_eps=1e-12, sd=0):
        super().__init__()

        self.blocks = nn.ModuleList([
            BlockVanilla(
                dim, num_heads, ff_dim, hidden_dropout_prob, attention_probs_dropout_prob,
                layer_norm_eps, sd) for _ in range(num_layers)])

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            x = block(x)

        return x
