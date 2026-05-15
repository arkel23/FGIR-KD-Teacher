"""PyTorch LRResNet

This started as a copy of https://github.com/pytorch/vision 'resnet.py' (BSD-3-Clause) with
additional dropout and dynamic global avg/max pool.

ResNeXt, SE-ResNeXt, SENet, and MXNet Gluon stem/downsample variants, tiered stems added by Ross Wightman

Copyright 2019, Ross Wightman

Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in:

'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929

`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270

`FlexiViT: One Model for All Patch Sizes`
    - https://arxiv.org/abs/2212.08013

The official jax code is released and available at
  * https://github.com/google-research/vision_transformer
  * https://github.com/google-research/big_vision

Acknowledgments:
  * The paper authors for releasing code and weights, thanks!
  * I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch
  * Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
  * Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020, Ross Wightman
"""
import math
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
from torchvision.ops import deform_conv2d
from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropBlock2d, DropPath, AvgPool2dSame, BlurPool2d, GroupNorm, LayerType, create_attn, \
    get_attn, get_act_layer, get_norm_layer, create_classifier, Mlp, trunc_normal_, use_fused_attn, RmsNorm
from timm.models._builder import build_model_with_cfg
from timm.models._manipulate import checkpoint_seq
from timm.models._registry import register_model, generate_default_cfgs, register_model_deprecations


__all__ = ['LRResNet', 'BasicBlock', 'Bottleneck']  # model_registry will add each entrypoint fn to this


def get_padding(kernel_size: int, stride: int, dilation: int = 1) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def create_aa(aa_layer: Type[nn.Module], channels: int, stride: int = 2, enable: bool = True) -> nn.Module:
    if not aa_layer or not enable:
        return nn.Identity()
    if issubclass(aa_layer, nn.AvgPool2d):
        return aa_layer(stride)
    else:
        return aa_layer(channels=channels, stride=stride)


class FSRCNN(nn.Module):
    def __init__(self, scale_factor, num_channels=3, d=56, s=12, m=4, norm=True):
        super(FSRCNN, self).__init__()
        ks_first = 5
        padding_first = ks_first // 2

        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d * scale_factor, kernel_size=ks_first, padding=padding_first),
            nn.PReLU(d * scale_factor)
        )


        ks_mid = 3
        padding_mid = ks_mid // 2

        self.mid_part = [nn.Conv2d(d * scale_factor, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend([
                nn.Conv2d(s, s, kernel_size=ks_mid, padding=padding_mid),
                nn.PReLU(s)
            ])
        self.mid_part.extend([nn.Conv2d(s, d * scale_factor, kernel_size=1), nn.PReLU(d * scale_factor)])
        self.mid_part = nn.Sequential(*self.mid_part)

        ks_last = 9
        padding_last = ks_last // 2
        self.last_part = nn.ConvTranspose2d(
            d * scale_factor, d, kernel_size=ks_last, stride=scale_factor,
            padding=padding_last, output_padding=scale_factor-1)

        if norm:
            self.norm = nn.BatchNorm2d(d)
            self.act = nn.ReLU(inplace=True)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)

        if hasattr(self, 'norm'):
            x = self.norm(x)
            x = self.act(x)

        return x


class DeformableDWConv2d(nn.Module):
    # https://arxiv.org/abs/1703.06211
    # https://arxiv.org/abs/1811.11168
    # https://github.com/developer0hye/PyTorch-Deformable-Convolution-v2/blob/main/dcn.py
    # https://pytorch.org/vision/main/generated/torchvision.ops.deform_conv2d.html
    def __init__(self,
                 in_channels,
                 channels_per_group,
                 kernel_size=1,
                 stride=2,
                 padding=0,
                 bias=False):
        super(DeformableDWConv2d, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding

        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size[0] * kernel_size[1],
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            # groups=1,
            bias=True
        )

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.dw_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * channels_per_group,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            groups=in_channels,
            bias=bias
        )

        # self.init_weights(rpds=True)

    @torch.jit.ignore
    def init_weights(self, rpds):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d) and 'offset' in n:
                nn.init.constant_(m.weight, 0.)
                nn.init.constant_(m.bias, 0.)
            elif rpds and isinstance(m, nn.Conv2d):
                print('Residual-preserving downsampling, initialized to near one values: ', n)
                nn.init.normal_(m.weight, mean=1.0, std=0.001)

    def forward(self, x):
        # h, w = x.shape[2:]
        # max_offset = max(h, w)/4.
        # print(x.shape, x)

        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
        # offset = torch.ones((1, 2, 2, 2)) to visualize the effects
        # print(offset.shape, offset)

        # op = (n - (k * d - 1) + 2p / s)
        x = deform_conv2d(
            input=x,
            offset=offset,
            weight=self.dw_conv.weight,
            bias=self.dw_conv.bias,
            padding=self.padding,
            stride=self.stride,
        )

        # print(x.shape, x)
        return x


class ResidualPreservingDWSConv(nn.Module):
    def __init__(self, in_channels, out_channels, 
    kernel_size=3, stride=2, padding=1, norm='batchnorm', activation='relu6',
    rpds=True, norm_act=True, deformable=True):
        super(ResidualPreservingDWSConv, self).__init__()

        channels_per_group = out_channels // in_channels

        # if in_channels == out_channels:
        #     stride = 1
            
        self.dwconv = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels * channels_per_group,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            groups=in_channels,
            bias=False,
        )

        self.norm_act = norm_act

        if activation == 'silu':
            self.activation = nn.SiLU(inplace=True)
        elif activation == 'relu6':
            self.activation = nn.ReLU6(inplace=True)
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)

        if norm == 'batchnorm':
            self.norm = nn.BatchNorm2d(in_channels * channels_per_group)
            self.norm2 = nn.BatchNorm2d(out_channels)

        self.pwconv = nn.Conv2d(
            in_channels * channels_per_group,
            out_channels,
            1,
            1,
            0,
            bias=False,
        )

        if stride != 1 and rpds:
            # residual preserving downsampling
            if deformable:
                self.rpds = DeformableDWConv2d(
                    in_channels=in_channels,
                    channels_per_group=channels_per_group,
                    kernel_size=1,
                    stride=(stride, stride),
                    padding=0,
                    bias=False,
                )
            else:
                self.rpds = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels * channels_per_group,
                    kernel_size=1,
                    stride=(stride, stride),
                    padding=0,
                    groups=in_channels,
                    bias=False,
                )
            # self.rpds_norm = nn.BatchNorm2d(in_channels * channels_per_group)

    def forward(self, x):
        # print(x.shape)
        # if hasattr(self, 'rpds'):
        res = x

        x = self.dwconv(x)
        # print(x.shape)

        if self.norm_act:
            if hasattr(self, 'norm'):
                x = self.norm(x)
            if hasattr(self, 'activation'):
                x = self.activation(x)
        else:
            if hasattr(self, 'activation'):
                x = self.activation(x)
            if hasattr(self, 'norm'):
                x = self.norm(x)

        x = self.pwconv(x)
        # print(x.shape)

        if self.norm_act:
            if hasattr(self, 'norm'):
                x = self.norm2(x)
            if hasattr(self, 'activation'):
                x = self.activation(x)
        else:
            if hasattr(self, 'activation'):
                x = self.activation(x)
            if hasattr(self, 'norm'):
                x = self.norm2(x)

        if hasattr(self, 'rpds'):
            # print(x.shape, res.shape)
            # x = x + self.rpds_norm(self.rpds(res))
            # print(x.shape, res.shape, self.rpds(res).shape)
            x = x + self.rpds(res)
        else:
            x = x + res

        return x


class ResidualPreservingConv(nn.Module):
    def __init__(self, in_channels, out_channels, 
    kernel_size=3, stride=2, padding=1, norm='batchnorm', activation='relu',
    rpds=True, deformable=True, drop_path=0.0):
        super(ResidualPreservingConv, self).__init__()

        channels_per_group = out_channels // in_channels
            
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            bias=False,
        )

        if norm == 'batchnorm':
            self.norm = nn.BatchNorm2d(out_channels)

        if activation == 'silu':
            self.activation = nn.SiLU(inplace=True)
        elif activation == 'relu6':
            self.activation = nn.ReLU6(inplace=True)
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)

        if rpds and (stride != 1 or in_channels != out_channels):
            # residual preserving downsampling
            if deformable:
                self.rpds = DeformableDWConv2d(
                    in_channels=in_channels,
                    channels_per_group=channels_per_group,
                    kernel_size=1,
                    stride=(stride, stride),
                    padding=0,
                    bias=False,
                )
            else:
                self.rpds = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels * channels_per_group,
                    kernel_size=1,
                    stride=(stride, stride),
                    padding=0,
                    groups=in_channels,
                    bias=False,
                )
            # self.rpds_norm = nn.BatchNorm2d(in_channels * channels_per_group)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # print(x.shape)
        # if hasattr(self, 'rpds'):
        res = x

        x = self.conv(x)
        # print(x.shape)

        if hasattr(self, 'norm'):
            x = self.norm(x)
        if hasattr(self, 'activation'):
            x = self.activation(x)

        if hasattr(self, 'rpds'):
            # print(x.shape, res.shape)
            # x = x + self.rpds_norm(self.rpds(res))
            # print(x.shape, res.shape, self.rpds(res).shape)
            x = self.drop_path(x) + self.rpds(res)
        else:
            x = self.drop_path(x) + res

        return x


class ResidualPreservingStem(nn.Module):
    def __init__(self, img_size=224, input_channels=3, hidden_size=32,
                 patch_size=2, conv_type='dws', rpds=True, deformable=True,
                 drop_path=0.0, norm='batchnorm', activation='relu'):
        super(ResidualPreservingStem, self).__init__()

        inc = input_channels

        if patch_size == 4:
            channels_in_list = [inc, hidden_size // 2, hidden_size]
            channels_out_list = [hidden_size // 2, hidden_size, hidden_size]
            stride_list = [2, 2, 1]

        elif patch_size == 2 and conv_type == 'dws':
            channels_in_list = [inc, hidden_size, hidden_size]
            channels_out_list = [hidden_size, hidden_size, hidden_size]
            stride_list = [2, 1, 1]

        elif patch_size == 2 and conv_type == 'conv':
            channels_in_list = [inc, hidden_size]
            channels_out_list = [hidden_size, hidden_size]
            stride_list = [1, 2]

        if conv_type == 'dws':
            self.conv3x3layers = nn.ModuleList([
                ResidualPreservingDWSConv(channels_in, channels_out,
                                          stride=stride, rpds=rpds,
                                          deformable=deformable)
                for (channels_in, channels_out, stride) in zip(channels_in_list, channels_out_list, stride_list)
            ])
        elif conv_type == 'conv':
            self.conv3x3layers = nn.ModuleList([
                ResidualPreservingConv(channels_in, channels_out,
                                       stride=stride, rpds=rpds,
                                       deformable=deformable, drop_path=drop_path)
                for (channels_in, channels_out, stride) in zip(channels_in_list, channels_out_list, stride_list)
            ])

        '''
        self.conv1x1 = nn.Conv2d(channels_out_list[-1], hidden_size, kernel_size=1,
                                 stride=1, padding=0, bias=False)
        if norm == 'batchnorm':
            self.norm = nn.BatchNorm2d(hidden_size)

        if activation == 'silu':
            self.activation = nn.SiLU(inplace=True)
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'relu6':
            self.activation = nn.ReLU6(inplace=True)

        # residual preserving module
        self.rp1x1_type = rp1x1_type

        if rp1x1_type == 'pwconv':
            self.rp1x1 = nn.Conv2d(
                in_channels=channels_out_list[-1], out_channels=hidden_size,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=channels_out_list[-1],
                bias=False,
            )

        elif rp1x1_type == 'image':
            self.rp1x1 = nn.Conv2d(
                in_channels=input_channels, out_channels=hidden_size,
                kernel_size=1,
                # kernel_size=(patch_size, patch_size),
                stride=(patch_size, patch_size),
                padding=0,
                groups=input_channels,
                bias=False,
            )
        '''

        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        # print(x.shape)

        # if hasattr(self, 'rp1x1') and self.rp1x1_type == 'image':
        #     res = x

        for layer in self.conv3x3layers: 
            x = layer(x)
            # print(x.shape)

        '''
        if hasattr(self, 'rp1x1') and self.rp1x1_type == 'pwconv':
            res = x

        x = self.conv1x1(x)

        if hasattr(self, 'norm'):
            x = self.norm(x)
        if hasattr(self, 'activation'):
            x = self.activation(x)

        if hasattr(self, 'rp1x1'):
            x = x + self.rp1x1(res)
        '''

        # x = rearrange(x, 'b d fh fw -> b (fh fw) d')
        # print(x.shape)

        return x


class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class TransformerParallelScalingBlock(nn.Module):
    """ Parallel ViT block (MLP & Attention in parallel)
    Based on:
      'Scaling Vision Transformers to 22 Billion Parameters` - https://arxiv.org/abs/2302.05442
    """
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            proj_dim_ratio: float = 0.25,
            mlp_ratio: float = 0.25,
            qkv_bias: bool = False,
            qk_norm: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: Optional[nn.Module] = None,
            head_dim: int = 64,
    ) -> None:
        super().__init__()
        proj_dim = int(proj_dim_ratio * dim)
        self.proj_dim = proj_dim

        num_heads = proj_dim // head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        assert proj_dim % num_heads == 0, 'proj_dim should be divisible by num_heads'

        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        mlp_hidden_dim = int(mlp_ratio * dim)
        in_proj_out_dim = mlp_hidden_dim + 3 * proj_dim

        if norm_layer == nn.BatchNorm1d:
            self.in_norm = nn.Sequential(
                Rearrange('b s d -> b d s'),
                norm_layer(dim),
                Rearrange('b d s -> b s d'),
            )
        else:
            self.in_norm = norm_layer(dim)
        self.in_proj = nn.Linear(dim, in_proj_out_dim, bias=qkv_bias)
        self.in_split = [mlp_hidden_dim] + [proj_dim] * 3
        if qkv_bias:
            self.register_buffer('qkv_bias', None)
            self.register_parameter('mlp_bias', None)
        else:
            self.register_buffer('qkv_bias', torch.zeros(3 * proj_dim), persistent=False)
            self.mlp_bias = nn.Parameter(torch.zeros(mlp_hidden_dim))

        if qk_norm and norm_layer == nn.BatchNorm1d:
            self.q_norm = nn.Sequential(
                Rearrange('b s nh dh -> b dh s nh'),
                nn.BatchNorm2d(self.head_dim),
                Rearrange('b dh s nh -> b s nh dh'),
            )
            self.k_norm = nn.Sequential(
                Rearrange('b s nh dh -> b dh s nh'),
                nn.BatchNorm2d(self.head_dim),
                Rearrange('b dh s nh -> b s nh dh'),
            )
        elif qk_norm:
            self.q_norm = norm_layer(self.head_dim)
            self.k_norm = norm_layer(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_out_proj = nn.Linear(proj_dim, dim)

        self.mlp_drop = nn.Dropout(proj_drop)
        self.mlp_act = act_layer()
        self.mlp_out_proj = nn.Linear(mlp_hidden_dim, dim)

        self.ls = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape

        # Combined MLP fc1 & qkv projections
        y = self.in_norm(x)
        if self.mlp_bias is not None:
            # Concat constant zero-bias for qkv w/ trainable mlp_bias.
            # Appears faster than adding to x_mlp separately
            y = F.linear(y, self.in_proj.weight, torch.cat((self.qkv_bias, self.mlp_bias)))
        else:
            y = self.in_proj(y)

        x_mlp, q, k, v = torch.split(y, self.in_split, dim=-1)

        # Dot product attention w/ qk norm
        q = self.q_norm(q.view(B, N, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(k.view(B, N, self.num_heads, self.head_dim)).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        if self.fused_attn and self.training:
            x_attn = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x_attn = attn @ v

        x_attn = x_attn.transpose(1, 2).reshape(B, N, self.proj_dim)
        x_attn = self.attn_out_proj(x_attn)

        # MLP activation, dropout, fc2
        x_mlp = self.mlp_act(x_mlp)
        x_mlp = self.mlp_drop(x_mlp)
        x_mlp = self.mlp_out_proj(x_mlp)

        # Add residual w/ drop path & layer scale applied
        y = self.drop_path(self.ls(x_attn + x_mlp))
        x = x + y
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            cardinality: int = 1,
            base_width: int = 64,
            reduce_first: int = 1,
            dilation: int = 1,
            first_dilation: Optional[int] = None,
            act_layer: Type[nn.Module] = nn.ReLU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            attn_layer: Optional[Type[nn.Module]] = None,
            aa_layer: Optional[Type[nn.Module]] = None,
            drop_block: Optional[Type[nn.Module]] = None,
            drop_path: Optional[nn.Module] = None,
    ):
        """
        Args:
            inplanes: Input channel dimensionality.
            planes: Used to determine output channel dimensionalities.
            stride: Stride used in convolution layers.
            downsample: Optional downsample layer for residual path.
            cardinality: Number of convolution groups.
            base_width: Base width used to determine output channel dimensionality.
            reduce_first: Reduction factor for first convolution output width of residual blocks.
            dilation: Dilation rate for convolution layers.
            first_dilation: Dilation rate for first convolution layer.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            attn_layer: Attention layer.
            aa_layer: Anti-aliasing layer.
            drop_block: Class for DropBlock layer.
            drop_path: Optional DropPath layer.
        """
        super(BasicBlock, self).__init__()

        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock does not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(
            inplanes, first_planes, kernel_size=3, stride=1 if use_aa else stride, padding=first_dilation,
            dilation=first_dilation, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act1 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=first_planes, stride=stride, enable=use_aa)

        self.conv2 = nn.Conv2d(
            first_planes, outplanes, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        if getattr(self.bn2, 'weight', None) is not None:
            nn.init.zeros_(self.bn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop_block(x)
        x = self.act1(x)
        x = self.aa(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            cardinality: int = 1,
            base_width: int = 64,
            reduce_first: int = 1,
            dilation: int = 1,
            first_dilation: Optional[int] = None,
            act_layer: Type[nn.Module] = nn.ReLU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            attn_layer: Optional[Type[nn.Module]] = None,
            aa_layer: Optional[Type[nn.Module]] = None,
            drop_block: Optional[Type[nn.Module]] = None,
            drop_path: Optional[nn.Module] = None,
    ):
        """
        Args:
            inplanes: Input channel dimensionality.
            planes: Used to determine output channel dimensionalities.
            stride: Stride used in convolution layers.
            downsample: Optional downsample layer for residual path.
            cardinality: Number of convolution groups.
            base_width: Base width used to determine output channel dimensionality.
            reduce_first: Reduction factor for first convolution output width of residual blocks.
            dilation: Dilation rate for convolution layers.
            first_dilation: Dilation rate for first convolution layer.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            attn_layer: Attention layer.
            aa_layer: Anti-aliasing layer.
            drop_block: Class for DropBlock layer.
            drop_path: Optional DropPath layer.
        """
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
            padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.bn2 = norm_layer(width)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act2 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=width, stride=stride, enable=use_aa)

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        if getattr(self.bn3, 'weight', None) is not None:
            nn.init.zeros_(self.bn3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop_block(x)
        x = self.act2(x)
        x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)

        return x


def downsample_conv(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        first_dilation: Optional[int] = None,
        norm_layer: Optional[Type[nn.Module]] = None,
) -> nn.Module:
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(*[
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False),
        norm_layer(out_channels)
    ])


def downsample_avg(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        first_dilation: Optional[int] = None,
        norm_layer: Optional[Type[nn.Module]] = None,
) -> nn.Module:
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(*[
        pool,
        nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        norm_layer(out_channels)
    ])


def downsample_dwc(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        first_dilation: Optional[int] = None,
        norm_layer: Optional[Type[nn.Module]] = None,
) -> nn.Module:
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(*[
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False, groups=in_channels),
    ])


def downsample_dwcn(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        first_dilation: Optional[int] = None,
        norm_layer: Optional[Type[nn.Module]] = None,
) -> nn.Module:
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(*[
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False, groups=in_channels),
        norm_layer(out_channels)
    ])


def downsample_deformabledwc(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        first_dilation: Optional[int] = None,
        norm_layer: Optional[Type[nn.Module]] = None,
) -> nn.Module:
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    channels_per_group = out_channels // in_channels

    return nn.Sequential(*[
        DeformableDWConv2d(
            in_channels, channels_per_group, kernel_size, stride=stride, padding=p),
    ])


def downsample_deformabledwcn(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        first_dilation: Optional[int] = None,
        norm_layer: Optional[Type[nn.Module]] = None,
) -> nn.Module:
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    channels_per_group = out_channels // in_channels

    return nn.Sequential(*[
        DeformableDWConv2d(
            in_channels, channels_per_group, kernel_size, stride=stride, padding=p),
        norm_layer(out_channels)
    ])


def drop_blocks(drop_prob: float = 0.):
    return [
        None, None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=1, gamma_scale=0.25) if drop_prob else None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=1, gamma_scale=0.5) if drop_prob else None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=1, gamma_scale=1.00) if drop_prob else None]


def make_blocks(
        block_fn: Union[BasicBlock, Bottleneck],
        channels: List[int],
        block_repeats: List[int],
        inplanes: int,
        reduce_first: int = 1,
        output_stride: int = 32,
        down_kernel_size: int = 1,
        residual_down: str = 'conv',
        drop_block_rate: float = 0.,
        drop_path_rate: float = 0.,
        **kwargs,
) -> Tuple[List[Tuple[str, nn.Module]], List[Dict[str, Any]]]:
    stages = []
    feature_info = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    net_stride = 2
    dilation = prev_dilation = 1
    for stage_idx, (planes, num_blocks, db) in enumerate(zip(channels, block_repeats, drop_blocks(drop_block_rate))):
        stage_name = f'layer{stage_idx + 1}'  # never liked this name, but weight compat requires it
        stride = 1 if stage_idx == 0 else 2
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = dict(
                in_channels=inplanes,
                out_channels=planes * block_fn.expansion,
                kernel_size=down_kernel_size,
                stride=stride,
                dilation=dilation,
                first_dilation=prev_dilation,
                norm_layer=kwargs.get('norm_layer'),
            )
            if residual_down == 'avg':
                downsample = downsample_avg(**down_kwargs)
            elif residual_down == 'dwc':
                downsample = downsample_dwc(**down_kwargs)
            elif residual_down == 'dwcn':
                downsample = downsample_dwcn(**down_kwargs)
            elif residual_down == 'deformabledwc':
                downsample = downsample_deformabledwc(**down_kwargs)
            elif residual_down == 'deformabledwcn':
                downsample = downsample_deformabledwcn(**down_kwargs)
            elif residual_down == 'conv':
                downsample = downsample_conv(**down_kwargs)

        block_kwargs = dict(reduce_first=reduce_first, dilation=dilation, drop_block=db, **kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            block_dpr = drop_path_rate * net_block_idx / (net_num_blocks - 1)  # stochastic depth linear decay rule
            blocks.append(block_fn(
                inplanes,
                planes,
                stride,
                downsample,
                first_dilation=prev_dilation,
                drop_path=DropPath(block_dpr) if block_dpr > 0. else None,
                **block_kwargs,
            ))
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1

        stages.append((stage_name, nn.Sequential(*blocks)))
        feature_info.append(dict(num_chs=inplanes, reduction=net_stride, module=stage_name))

    return stages, feature_info


# https://github.com/EIFY/mup-vit
# Taken from https://github.com/lucidrains/vit-pytorch, likely ported from https://github.com/google-research/big_vision/
def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class LearnedPositionalEmbedding1D(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))

    def forward(self, x):
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        x = x + self.pos_embedding
        return x


class LRResNet(nn.Module):
    """LRResNet / ResNeXt / SE-ResNeXt / SE-Net

    This class implements all variants of LRResNet, ResNeXt, SE-ResNeXt, and SENet that
      * have > 1 stride in the 3x3 conv layer of bottleneck
      * have conv-bn-act ordering

    This LRResNet impl supports a number of stem and downsample options based on the v1c, v1d, v1e, and v1s
    variants included in the MXNet Gluon LRResNetV1b model. The C and D variants are also discussed in the
    'Bag of Tricks' paper: https://arxiv.org/pdf/1812.01187. The B variant is equivalent to torchvision default.

    LRResNet variants (the same modifications can be used in SE/ResNeXt models as well):
      * normal, b - 7x7 stem, stem_width = 64, same as torchvision LRResNet, NVIDIA LRResNet 'v1.5', Gluon v1b
      * c - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64)
      * d - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64), average pool in downsample
      * e - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128), average pool in downsample
      * s - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128)
      * t - 3 layer deep 3x3 stem, stem width = 32 (24, 48, 64), average pool in downsample
      * tn - 3 layer deep 3x3 stem, stem width = 32 (24, 32, 64), average pool in downsample

    ResNeXt
      * normal - 7x7 stem, stem_width = 64, standard cardinality and base widths
      * same c,d, e, s variants as LRResNet can be enabled

    SE-ResNeXt
      * normal - 7x7 stem, stem_width = 64
      * same c, d, e, s variants as LRResNet can be enabled

    SENet-154 - 3 layer deep 3x3 stem (same as v1c-v1s), stem_width = 64, cardinality=64,
        reduction by 2 on width of first bottleneck convolution, 3x3 downsample convs after first block
    """

    def __init__(
            self,
            block: Union[BasicBlock, Bottleneck],
            layers: List[int],
            num_classes: int = 1000,
            in_chans: int = 3,
            output_stride: int = 32,
            global_pool: str = 'avg',
            cardinality: int = 1,
            base_width: int = 64,
            stem_width: int = 32,
            stem_type: str = '',
            replace_stem_pool: bool = False,
            block_reduce_first: int = 1,
            down_kernel_size: int = 1,
            residual_down: str = 'conv',
            act_layer: LayerType = nn.ReLU,
            norm_layer: LayerType = nn.BatchNorm2d,
            aa_layer: Optional[Type[nn.Module]] = None,
            drop_rate: float = 0.0,
            drop_path_rate: float = 0.,
            drop_block_rate: float = 0.,
            zero_init_last: bool = True,
            block_args: Optional[Dict[str, Any]] = None,
            transformer_blocks: int = 0,
            mlp_ratio: float = 0.25,
            img_size: int = 224,
            pos_embedding_type: str = 'sin2d',
            drop_cls_token: bool = False,
            init_values: float = 1e-5,
            inter_feats: bool = False,
            ulr: bool = False,
            sr: int = None,
            norm_transformer: LayerType = nn.BatchNorm1d,
    ):
        """
        Args:
            block (nn.Module): class for the residual block. Options are BasicBlock, Bottleneck.
            layers (List[int]) : number of layers in each block
            num_classes (int): number of classification classes (default 1000)
            in_chans (int): number of input (color) channels. (default 3)
            output_stride (int): output stride of the network, 32, 16, or 8. (default 32)
            global_pool (str): Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax' (default 'avg')
            cardinality (int): number of convolution groups for 3x3 conv in Bottleneck. (default 1)
            base_width (int): bottleneck channels factor. `planes * base_width / 64 * cardinality` (default 64)
            stem_width (int): number of channels in stem convolutions (default 64)
            stem_type (str): The type of stem (default ''):
                * 'patch': 2x2 non-overlapping (vit style, also used in Swin and CNX)
                * 'patch_overlap': 2x2 overlap with stride 1
                * 'patch_dw': 2x2 non-overlapping with dw conv
                * 'patch_dw_overlap': 2x2 with 2x2 stride and padding=2
                * '', default - a single 7x7 conv with a width of stem_width
                * 'deep' - three 3x3 convolution layers of widths stem_width, stem_width, stem_width * 2
                * 'deep_tiered' - three 3x3 conv layers of widths stem_width//4 * 3, stem_width, stem_width * 2
            replace_stem_pool (bool): replace stem max-pooling layer with a 3x3 stride-2 convolution
            block_reduce_first (int): Reduction factor for first convolution output width of residual blocks,
                1 for all archs except senets, where 2 (default 1)
            down_kernel_size (int): kernel size of residual block downsample path,
                1x1 for most, 3x3 for senets (default: 1)
            avg_down (bool): use avg pooling for projection skip connection between stages/downsample (default False) ->
            changed to residual_down
            residual_down (str): use avg pooling for projection skip connection between stages/downsample (default conv)
                * avg: avg + conv
                * conv: conv with stride
                * dwc: dw conv
                * dwcn: dw conv with norm
            act_layer (str, nn.Module): activation layer
            norm_layer (str, nn.Module): normalization layer
            aa_layer (nn.Module): anti-aliasing layer
            drop_rate (float): Dropout probability before classifier, for training (default 0.)
            drop_path_rate (float): Stochastic depth drop-path rate (default 0.)
            drop_block_rate (float): Drop block rate (default 0.)
            zero_init_last (bool): zero-init the last weight in residual path (usually last BN affine weight)
            block_args (dict): Extra kwargs to pass through to block module
        """
        super(LRResNet, self).__init__()
        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        
        act_layer = get_act_layer(act_layer)
        norm_layer = get_norm_layer(norm_layer)

        # Stem
        FIRST_CONV_FEATURES = 32
        deep_stem = 'deep' in stem_type
        if stem_type == 'rps':
            inplanes = 32
        else:
            inplanes = FIRST_CONV_FEATURES
            # inplanes = stem_width * 2 if deep_stem else FIRST_CONV_FEATURES

        if sr:
            self.conv1 = FSRCNN(sr, in_chans, inplanes)
            self.feature_info = [dict(num_chs=inplanes, reduction=1/sr, module='conv1')]

        elif stem_type == 'rpcs_dwc':
            self.conv1 = ResidualPreservingStem(
                img_size, in_chans, inplanes, patch_size=2, conv_type='conv',
                deformable=False)
            self.feature_info = [dict(num_chs=inplanes, reduction=2, module='conv1')]
        elif stem_type == 'rpcs':
            self.conv1 = ResidualPreservingStem(
                img_size, in_chans, inplanes, patch_size=2, conv_type='conv',
                deformable=True)
            self.feature_info = [dict(num_chs=inplanes, reduction=2, module='conv1')]
        elif stem_type == 'rps':
            self.conv1 = ResidualPreservingStem(
                img_size, in_chans, inplanes, patch_size=2, conv_type='dws',
                deformable=True)
            self.feature_info = [dict(num_chs=inplanes, reduction=2, module='conv1')]
        elif stem_type == 'rps_dwc':
            self.conv1 = ResidualPreservingStem(
                img_size, in_chans, inplanes, patch_size=2, conv_type='dws',
                deformable=False)
            self.feature_info = [dict(num_chs=inplanes, reduction=2, module='conv1')]

        elif 'patch' in stem_type or 'dw' in stem_type:
            ps = 4 if 'overlap' in stem_type else 2
            padding = 1 if 'overlap' in stem_type else 0
            # equivalent flops to traditional patch conv
            dw_chans = int((inplanes * (ps ** 2)) / (in_chans * (ps ** 2) + inplanes))

            if stem_type in ('patch', 'patch_overlap'):
                self.conv1 = nn.Sequential(
                    nn.Conv2d(in_chans, inplanes, kernel_size=ps, stride=2, padding=padding, bias=False),
                    norm_layer(inplanes),
                    act_layer(inplace=True),
                )
            elif stem_type in ('patch_dw', 'patch_dw_overlap'):
                self.conv1 = nn.Sequential(
                    nn.Conv2d(in_chans, in_chans * dw_chans, kernel_size=ps, stride=2, padding=padding, bias=False, groups=in_chans),
                    norm_layer(in_chans * dw_chans),
                    act_layer(inplace=True),
                    nn.Conv2d(in_chans * dw_chans, inplanes, kernel_size=1, stride=1, padding=0),
                    norm_layer(inplanes),
                    act_layer(inplace=True),
                )

            self.feature_info = [dict(num_chs=inplanes, reduction=2, module='conv1')]

        else:
            if deep_stem:
                stem_chs = (stem_width, stem_width)
                if 'tiered' in stem_type:
                    stem_chs = (3 * (stem_width // 4), stem_width)
                self.conv1 = nn.Sequential(*[
                    nn.Conv2d(in_chans, stem_chs[0], 3, stride=2, padding=1, bias=False),
                    norm_layer(stem_chs[0]),
                    act_layer(inplace=True),
                    nn.Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False),
                    norm_layer(stem_chs[1]),
                    act_layer(inplace=True),
                    nn.Conv2d(stem_chs[1], inplanes, 3, stride=1, padding=1, bias=False)])
            else:
                self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=3, stride=2, padding=1, bias=False)
            self.bn1 = norm_layer(inplanes)
            self.act1 = act_layer(inplace=True)
            self.feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]

        # Feature Blocks
        # OUTPUT_FEATURES = 512
        # channels = [32, 64, 128, 256, OUTPUT_FEATURES]

        # head_dim = 66
        head_dim = 64
        # OUTPUT_FEATURES = 528
        OUTPUT_FEATURES = 512
        # channels = [33, 66, 132, 264, OUTPUT_FEATURES]
        # channels = [66, 66, 132, 264, OUTPUT_FEATURES]
        channels = [64, 64, 128, 256, OUTPUT_FEATURES]

        # if stem_type == 'rps':
        #     head_dim = 60
        #     OUTPUT_FEATURES = 480
        #     channels = [30, 60, 120, 240, OUTPUT_FEATURES]

        # elif stem_type == 'rps':
        #     OUTPUT_FEATURES = 576
        #     channels = [36, 72, 144, 288, OUTPUT_FEATURES]
        # else:
        #     head_dim = 64
        #     OUTPUT_FEATURES = 512
        #     channels = [32, 64, 128, 256, OUTPUT_FEATURES]

        stage_modules, stage_feature_info = make_blocks(
            block,
            channels,
            layers,
            inplanes,
            cardinality=cardinality,
            base_width=base_width,
            output_stride=output_stride,
            reduce_first=block_reduce_first,
            residual_down=residual_down,
            down_kernel_size=down_kernel_size,
            act_layer=act_layer,
            norm_layer=norm_layer,
            aa_layer=aa_layer,
            drop_block_rate=drop_block_rate,
            drop_path_rate=drop_path_rate,
            **block_args,
        )
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)

        self.num_features = OUTPUT_FEATURES * block.expansion

        # transformer blocks
        if transformer_blocks:
            if block.expansion > 1:
                mlp_ratio = mlp_ratio / block.expansion
                pool_kernels = [16, 8, 4, 2]
                spr_ratios = [1, 1, 1, 1]

            else:
                pool_kernels = [4, 2, 1, 1]
                spr_ratios = [4, 4, 4, 2]

            fh = math.floor(img_size / output_stride)
            seq_len = int(fh ** 2)

            if inter_feats:
                if block.expansion > 1:
                    self.space_to_depth = nn.ModuleList([nn.Sequential(
                        nn.AvgPool2d(pool_k, pool_k, padding=0),
                        Rearrange('b d (fh r1) (fw r2) -> b (d r1 r2) (fh fw)', r1=r, r2=r),
                        nn.Conv1d(
                            in_channels=(d * block.expansion * r * r),
                            out_channels=self.num_features,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            groups=(d * block.expansion * r * r),
                            bias=False
                        ),
                        Rearrange('b d s -> b s d'),
                    ) for pool_k, d, r in zip(pool_kernels, channels, spr_ratios)])
                else:
                    self.space_to_depth = nn.ModuleList([nn.Sequential(
                        nn.AvgPool2d(pool_k, pool_k, padding=0),
                        Rearrange('b d (fh r1) (fw r2) -> b (fh fw) (d r1 r2)', r1=r, r2=r),
                        nn.Linear(d * block.expansion * r * r, self.num_features),
                    ) for pool_k, d, r in zip(pool_kernels, channels, spr_ratios)])

                self.space_to_depth.append(Rearrange('b d fh fw -> b (fh fw) d'))

            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.num_features))

            # Positional embedding
            if pos_embedding_type == 'learned':
                self.positional_embedding = LearnedPositionalEmbedding1D(
                    seq_len, self.num_features)
            elif pos_embedding_type == 'sin2d':
                self.register_buffer("sin2d_embedding", posemb_sincos_2d(fh, fh, self.num_features))

            # Transformer encoder
            self.encoder = nn.ModuleList([TransformerParallelScalingBlock(
                self.num_features,
                mlp_ratio,
                mlp_ratio,
                qkv_bias=True,
                qk_norm=True,
                init_values=init_values,
                drop_path=drop_path_rate,
                norm_layer=norm_transformer,
            ) for i in range(transformer_blocks)])

            self.encoder_norm = nn.LayerNorm(self.num_features, eps=1e-6)

            self.global_pool = nn.Identity()
            self.fc = nn.Identity()

            if self.num_classes:
                self.classifier_cls = True
                self.fc = nn.Linear(self.num_features, self.num_classes)

            if drop_cls_token:
                self.drop_class_token = True

        else:
            # Head (Pooling and Classifier)
            self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

        self.init_weights(zero_init_last=zero_init_last, residual_down=residual_down)

    @torch.jit.ignore
    def init_weights(self, zero_init_last: bool = True, residual_down: str = 'conv'):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d) and 'offset' in n:
                continue
            elif isinstance(m, nn.Conv2d) and 'dwc' in residual_down and 'downsample' in n:
                print('Residual-preserving downsampling, initialized to near one values: ', n)
                nn.init.normal_(m.weight, mean=1.0, std=0.001)
                # nn.init.normal_(m.bias, mean=0.0, std=0.001)
            elif isinstance(m, nn.Conv2d) and 'rp' in n:
                print('Residual-preserving downsampling, initialized to near one values: ', n)
                nn.init.normal_(m.weight, mean=1.0, std=0.001)
                # nn.init.normal_(m.bias, mean=0.0, std=0.001)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if zero_init_last:
            for m in self.modules():
                if hasattr(m, 'zero_init_last'):
                    m.zero_init_last()
        if hasattr(self, 'cls_token'):
            nn.init.normal_(self.cls_token, std=1e-6)
        if hasattr(self, 'positional_embedding'):
            trunc_normal_(self.positional_embedding.pos_embedding, std=.02)

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False):
        matcher = dict(stem=r'^conv1|bn1|maxpool', blocks=r'^layer(\d+)' if coarse else r'^layer(\d+)\.(\d+)')
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self, name_only: bool = False):
        return 'fc' if name_only else self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def prepare_inter_feats(self, x, inter_feats, level=0):
        inter = self.space_to_depth[level](x)
        if hasattr(self, 'sin2d_embedding'):
            inter = inter + self.sin2d_embedding
        inter_feats.append(inter)
        return 0

    def forward_cnn_features(self, x: torch.Tensor) -> torch.Tensor:
        inter_feats = []

        # print(x.shape)

        b = x.shape[0]
        x = self.conv1(x)

        # missing from original set of runs
        if hasattr(self, 'bn1'):
            x = self.bn1(x)
            x = self.act1(x)

        # if hasattr(self, 'inter_pool'):
        #     inter_feats.append(rearrange(self.inter_pool(x), 'b d fh fw -> b (fh fw) d'))

        # print(x.shape)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq([self.layer1, self.layer2, self.layer3, self.layer4, self.layer5], x, flatten=True)
        else:
            x = self.layer1(x)
            # print(x.shape)
            if hasattr(self, 'space_to_depth'):
                self.prepare_inter_feats(x, inter_feats, level=0)
                # inter_feats.append(rearrange(self.inter_pool(x), 'b d fh fw -> b (fh fw) d'))
            # print(x.shape)

            x = self.layer2(x)
            if hasattr(self, 'space_to_depth'):
                self.prepare_inter_feats(x, inter_feats, level=1)

            # if hasattr(self, 'space_to_depth'):
            #     inter = self.space_to_depth[1](x)
            #     inter = rearrange(x, 'b d fh fw -> b (fh fw) d')
            #     if hasattr(self, 'sin2d_embedding'):
            #         inter = inter + self.sin2d_embedding
            #     inter_feats.append(inter)
            # if hasattr(self, 'inter_pool'):
            #     inter_feats.append(rearrange(self.inter_pool(x), 'b d fh fw -> b (fh fw) d'))
            # print(x.shape)

            x = self.layer3(x)
            if hasattr(self, 'space_to_depth'):
                self.prepare_inter_feats(x, inter_feats, level=2)

            # if hasattr(self, 'space_to_depth'):
            #     inter = self.space_to_depth[2](x)
            #     inter = rearrange(x, 'b d fh fw -> b (fh fw) d')
            #     if hasattr(self, 'sin2d_embedding'):
            #         inter = inter + self.sin2d_embedding
            #     inter_feats.append(inter)

            # if hasattr(self, 'inter_pool'):
            #     # inter_feats.append(rearrange(self.inter_pool(x), 'b d fh fw -> b (fh fw) d'))
            #     inter_feats.append(self.inter_pool(x))
            # print(x.shape)

            x = self.layer4(x)
            if hasattr(self, 'space_to_depth'):
                self.prepare_inter_feats(x, inter_feats, level=3)

            # if hasattr(self, 'space_to_depth'):
            #     inter = self.space_to_depth[3](x)
            #     inter = rearrange(x, 'b d fh fw -> b (fh fw) d')
            #     if hasattr(self, 'sin2d_embedding'):
            #         inter = inter + self.sin2d_embedding
            #     inter_feats.append(inter)
            # if hasattr(self, 'inter_pool'):
            #     # inter_feats.append(rearrange(self.inter_pool(x), 'b d fh fw -> b (fh fw) d'))
            #     inter_feats.append(self.inter_pool(x))
            # print(x.shape)

            x = self.layer5(x)
            if hasattr(self, 'space_to_depth'):
                self.prepare_inter_feats(x, inter_feats, level=4)

            # if hasattr(self, 'space_to_depth'):
            #     inter = self.space_to_depth[4](x)
            #     inter = rearrange(x, 'b d fh fw -> b (fh fw) d')
            #     if hasattr(self, 'sin2d_embedding'):
            #         inter = inter + self.sin2d_embedding
            #     inter_feats.append(inter)

            # if hasattr(self, 'inter_pool'):
            #     # inter_feats.append(rearrange(self.inter_pool(x), 'b d fh fw -> b (fh fw) d'))
            #     inter_feats.append(self.inter_pool(x))
            # print(x.shape)

        return x, inter_feats

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x, inter_feats = self.forward_cnn_features(x)

        if hasattr(self, 'encoder'):
            if hasattr(self, 'space_to_depth'):
                seq_len = inter_feats[-1].shape[1]
                x = torch.cat(inter_feats, dim=1)
            # if hasattr(self, 'inter_pool'):
            #     # x = torch.cat(inter_feats, dim=-1)
            #     x = torch.cat(inter_feats, dim=1)
            #     x = self.clf(x)
            #     x = rearrange(x, 'b d fh fw -> b (fh fw) d')
            else:
                x = rearrange(x, 'b d fh fw -> b (fh fw) d')

            if hasattr(self, 'sin2d_embedding') and not hasattr(self, 'space_to_depth'):
                x = x + self.sin2d_embedding

            if hasattr(self, 'cls_token'):
                cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=x.shape[0])
                x = torch.cat((cls_tokens, x), dim=1)

            if hasattr(self, 'positional_embedding'):
                x = self.positional_embedding(x)

            for i, blk in enumerate(self.encoder):
                x = blk(x)
                if hasattr(self, 'space_to_depth') and (i + 1) == (len(self.encoder) - 1):
                    x = torch.cat([x[:, :1], x[:, -seq_len:]], dim=1)

            if hasattr(self, 'drop_class_token'):
                x = x[:, 1:]

            x = self.encoder_norm(x)

        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        if hasattr(self, 'encoder') and hasattr(self, 'classifier_cls'):
            return self.fc(x[:, 0])

        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        return x if pre_logits else self.fc(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_lrnet(variant, pretrained: bool = False, **kwargs) -> LRResNet:
    return build_model_with_cfg(LRResNet, variant, pretrained, **kwargs)


# equivalent to resnet14
@register_model
def sr2xlrnet14(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-14 model.
    """
    model_args = dict(block=BasicBlock, layers=[1, 2, 1, 1, 1], sr=2)
    return _create_lrnet('sr2xlrnet14', pretrained, **dict(model_args, **kwargs))


@register_model
def sr4xlrnet14(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-14 model.
    """
    model_args = dict(block=BasicBlock, layers=[1, 2, 1, 1, 1], sr=4)
    return _create_lrnet('sr4xlrnet14', pretrained, **dict(model_args, **kwargs))


@register_model
def sr8xlrnet14(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-14 model.
    """
    model_args = dict(block=BasicBlock, layers=[1, 2, 1, 1, 1], sr=8)
    return _create_lrnet('sr8xlrnet14', pretrained, **dict(model_args, **kwargs))


@register_model
def sr2xlrnet18(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-18 model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 1, 1], sr=2)
    return _create_lrnet('sr2xlrnet18', pretrained, **dict(model_args, **kwargs))


@register_model
def sr4xlrnet18(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-18 model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 1, 1], sr=4)
    return _create_lrnet('sr4xlrnet18', pretrained, **dict(model_args, **kwargs))


@register_model
def sr8xlrnet18(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-18 model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 1, 1], sr=8)
    return _create_lrnet('sr8xlrnet18', pretrained, **dict(model_args, **kwargs))


@register_model
def sr8xlrnet34(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-34 model.
    """
    model_args = dict(block=BasicBlock, layers=[3, 3, 3, 4, 3], sr=8)
    return _create_lrnet('sr8xlrnet34', pretrained, **dict(model_args, **kwargs))


# equivalent to resnet14
@register_model
def lrnet14(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-14 model.
    """
    model_args = dict(block=BasicBlock, layers=[1, 2, 1, 1, 1])
    return _create_lrnet('lrnet14', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet14t2(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-14 model.
    """
    model_args = dict(block=BasicBlock, layers=[1, 2, 1, 1, 1],
                      transformer_blocks=2)
    return _create_lrnet('lrnet14t2', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet14ift2(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-14 model.
    """
    model_args = dict(block=BasicBlock, layers=[1, 2, 1, 1, 1],
                      transformer_blocks=2, inter_feats=True)
    return _create_lrnet('lrnet14ift2', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet14d(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-14 model.
    """
    model_args = dict(block=BasicBlock, layers=[1, 2, 1, 1, 1],
                      stem_width=18, stem_type='deep', residual_down='avg')
    return _create_lrnet('lrnet14d', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet14dwr(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-14 Residual Preserving Stem + Residual model.
    """
    model_args = dict(block=BasicBlock, layers=[1, 2, 1, 1, 1],
                      residual_down='dwc')
    return _create_lrnet('lrnet14dwr', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet14ddwr(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-14 Residual Preserving Stem + Residual model.
    """
    model_args = dict(block=BasicBlock, layers=[1, 2, 1, 1, 1],
                      residual_down='deformabledwc')
    return _create_lrnet('lrnet14ddwr', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet14rpdws(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-14-Residual Preserving Stem model.
    """
    model_args = dict(block=BasicBlock, layers=[1, 2, 1, 1, 1], stem_type='rps_dwc')
    return _create_lrnet('lrnet14rpdws', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet14rpcs(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-14-Residual Preserving Stem model.
    """
    model_args = dict(block=BasicBlock, layers=[1, 2, 1, 1, 1], stem_type='rpcs')
    return _create_lrnet('lrnet14rpcs', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet14rpcdws(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-14-Residual Preserving Stem model.
    """
    model_args = dict(block=BasicBlock, layers=[1, 2, 1, 1, 1], stem_type='rpcs_dwc')
    return _create_lrnet('lrnet14rpcdws', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet14rps(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-14-Residual Preserving Stem model.
    """
    model_args = dict(block=BasicBlock, layers=[1, 2, 1, 1, 1], stem_type='rps')
    return _create_lrnet('lrnet14rps', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet14rpsdwr(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-14 Residual Preserving Stem + Residual model.
    """
    model_args = dict(block=BasicBlock, layers=[1, 2, 1, 1, 1], stem_type='rpcs_dwc',
                      residual_down='dwc')
    return _create_lrnet('lrnet14rpsdwr', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet14rpsddwr(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-14 Residual Preserving Stem + Residual model.
    """
    model_args = dict(block=BasicBlock, layers=[1, 2, 1, 1, 1], stem_type='rpcs_dwc',
                      residual_down='deformabledwc')
    return _create_lrnet('lrnet14rpsddwr', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet14rpst2(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-14 Residual Preserving Stem + Residual model.
    """
    model_args = dict(block=BasicBlock, layers=[1, 2, 1, 1, 1], stem_type='rpcs_dwc',
                      transformer_blocks=2)
    return _create_lrnet('lrnet14rpst2', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet14rpst2ddwr(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-14 Residual Preserving Stem + Residual model.
    """
    model_args = dict(block=BasicBlock, layers=[1, 2, 1, 1, 1], stem_type='rpcs_dwc',
                      residual_down='deformabledwc', transformer_blocks=2)
    return _create_lrnet('lrnet14rpst2ddwr', pretrained, **dict(model_args, **kwargs))


# equivalent to resnet18
@register_model
def lrnet18(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-18 model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 1, 1])
    return _create_lrnet('lrnet18', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet18d(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-18 model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 1, 1],
                      stem_width=18, stem_type='deep', residual_down='avg')
    return _create_lrnet('lrnet18d', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet18ddwr(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-18 Residual Preserving Stem + Residual model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 1, 1],
                      residual_down='deformabledwc')
    return _create_lrnet('lrnet18ddwr', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet18rps(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-18-Residual Preserving Stem model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 1, 1], stem_type='rps')
    return _create_lrnet('lrnet18rps', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet18rpsddwr(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-18 Residual Preserving Stem + Residual model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 1, 1], stem_type='rpcs_dwc',
                      residual_down='deformabledwc')
    return _create_lrnet('lrnet18rpsddwr', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet18rpst2(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-18 Residual Preserving Stem + Residual model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 1, 1], stem_type='rps',
                      transformer_blocks=2)
    return _create_lrnet('lrnet18rpst2', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet18rpst2ddwr(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-18 Residual Preserving Stem + Residual model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 1, 1], stem_type='rps',
                      residual_down='deformabledwc', transformer_blocks=2)
    return _create_lrnet('lrnet18rpst2ddwr', pretrained, **dict(model_args, **kwargs))


# equivalent to resnet34 in flops
@register_model
def lrnet26(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-26 model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 2, 2, 1, 1])
    return _create_lrnet('lrnet26', pretrained, **dict(model_args, **kwargs))


# equivalent to resnet34 in flops
@register_model
def lrnet24(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-34 model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 3, 2])
    return _create_lrnet('lrnet24', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet28(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-28 model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 4, 3])
    return _create_lrnet('lrnet28', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet34(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-34 model.
    """
    model_args = dict(block=BasicBlock, layers=[3, 3, 3, 4, 3])
    return _create_lrnet('lrnet34', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet34t2(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-34 model.
    """
    model_args = dict(block=BasicBlock, layers=[3, 3, 3, 4, 3],
                      transformer_blocks=2)
    return _create_lrnet('lrnet34t2', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet34ift2(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-34 model.
    """
    model_args = dict(block=BasicBlock, layers=[3, 3, 3, 4, 3],
                      transformer_blocks=2, inter_feats=True)
    return _create_lrnet('lrnet34ift2', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet34d(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-34 model.
    """
    model_args = dict(block=BasicBlock, layers=[3, 3, 3, 4, 3],
                      stem_width=18, stem_type='deep', residual_down='avg')
    return _create_lrnet('lrnet34d', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet34rps(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-34-Residual Preserving Stem model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 3, 4, 5, 2], stem_type='rps')
    return _create_lrnet('lrnet34rps', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet34rpsddwr(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-34 Residual Preserving Stem + Residual model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 3, 4, 5, 2], stem_type='rpcs_dwc',
                      residual_down='deformabledwc')
    return _create_lrnet('lrnet34rpsddwr', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet34rpst2(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-34 Residual Preserving Stem + Residual model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 3, 4, 5, 2], stem_type='rpcs_dwc',
                      transformer_blocks=2)
    return _create_lrnet('lrnet34rpst2', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet34rpst2ddwr(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-34 Residual Preserving Stem + Residual model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 3, 4, 5, 2], stem_type='rpcs_dwc',
                      residual_down='deformabledwc', transformer_blocks=2)
    return _create_lrnet('lrnet34rpst2ddwr', pretrained, **dict(model_args, **kwargs))


# equivalent to resnet35
@register_model
def lrnet35(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-35 model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 2, 2, 3, 2])
    return _create_lrnet('lrnet35', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet50(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-50 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 3, 3, 4, 3])
    return _create_lrnet('lrnet50', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet50t2(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-50 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 3, 3, 4, 3],
                      transformer_blocks=2)
    return _create_lrnet('lrnet50t2', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet50ift2(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-50 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 3, 3, 4, 3],
                      transformer_blocks=2, inter_feats=True)
    return _create_lrnet('lrnet50ift2', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet50d(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-50 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 3, 3, 4, 3],
                      stem_width=18, stem_type='deep', residual_down='avg')
    return _create_lrnet('lrnet50d', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet50rps(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-50-Residual Preserving Stem model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 3, 4, 5, 2], stem_type='rps')
    return _create_lrnet('lrnet50rps', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet50rpsddwr(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-50 Residual Preserving Stem + Residual model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 3, 4, 5, 2], stem_type='rpcs_dwc',
                      residual_down='deformabledwc')
    return _create_lrnet('lrnet50rpsddwr', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet50rpst2(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-50 Residual Preserving Stem + Residual model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 3, 4, 5, 2], stem_type='rpcs_dwc',
                      transformer_blocks=2)
    return _create_lrnet('lrnet50rpst2', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet50rpst2ddwr(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-50 Residual Preserving Stem + Residual model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 3, 4, 5, 2], stem_type='rpcs_dwc',
                      residual_down='deformabledwc', transformer_blocks=2)
    return _create_lrnet('lrnet50rpst2ddwr', pretrained, **dict(model_args, **kwargs))


# equivalent to resnet101
@register_model
def lrnet80(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-80 model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 2, 2, 18, 2])
    return _create_lrnet('lrnet80', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet74(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-74 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 3, 3, 12, 3])
    return _create_lrnet('lrnet74', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet101(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-101 model.
    """
    model_args = dict(block=Bottleneck, layers=[4, 4, 4, 17, 4])
    return _create_lrnet('lrnet101', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet101ift2(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-101 model.
    """
    model_args = dict(block=Bottleneck, layers=[4, 4, 4, 17, 4],
                      transformer_blocks=2, inter_feats=True)
    return _create_lrnet('lrnet101ift2', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet101d(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-101 model.
    """
    model_args = dict(block=Bottleneck, layers=[4, 4, 4, 17, 4],
                      stem_width=18, stem_type='deep', residual_down='avg')
    return _create_lrnet('lrnet101d', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet101rps(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-101-Residual Preserving Stem model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 3, 4, 22, 2], stem_type='rps')
    return _create_lrnet('lrnet101rps', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet101rpsddwr(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-101 Residual Preserving Stem + Residual model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 3, 4, 22, 2], stem_type='rpcs_dwc',
                      residual_down='deformabledwc')
    return _create_lrnet('lrnet101rpsddwr', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet101rpst2(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-101 Residual Preserving Stem + Residual model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 3, 4, 22, 2], stem_type='rpcs_dwc',
                      transformer_blocks=2)
    return _create_lrnet('lrnet101rpst2', pretrained, **dict(model_args, **kwargs))


@register_model
def lrnet101rpst2ddwr(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-101 Residual Preserving Stem + Residual model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 3, 4, 22, 2], stem_type='rpcs_dwc',
                      residual_down='deformabledwc', transformer_blocks=2)
    return _create_lrnet('lrnet101rpst2ddwr', pretrained, **dict(model_args, **kwargs))
