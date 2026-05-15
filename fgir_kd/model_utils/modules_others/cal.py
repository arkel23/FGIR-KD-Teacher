'''
https://github.com/raoyongming/CAL/tree/master/fgvc
https://github.com/raoyongming/CAL/blob/master/fgvc/train_distributed.py
https://github.com/raoyongming/CAL/blob/master/fgvc/models/cal.py
https://github.com/raoyongming/CAL/blob/master/fgvc/infer.py
https://github.com/raoyongming/CAL/blob/master/fgvc/utils.py
'''
import math
import random
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat

from .crd import CRDLoss


EPSILON = 1e-6


# augment function
def batch_augment(images, attention_map, mode='crop', theta=0.5, padding_ratio=0.1, percent_max=True, kur_adjust=False):
    batches, _, imgH, imgW = images.size()

    if mode == 'crop':
        crop_images = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                if percent_max:
                    theta_c = random.uniform(*theta) * atten_map.max()
                else:
                    theta_c = random.uniform(*theta) * atten_map.mean()
            else:
                if percent_max:
                    theta_c = theta * atten_map.max()
                else:
                    theta_c = theta * atten_map.mean()

            # 0 / 1 mask based on if attention at x,y is higher than max value * threshold percentage
            crop_mask = F.upsample_bilinear(atten_map, size=(imgH, imgW)) >= theta_c

            # x, y indices for 1 values in mask
            nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])

            # select highest/min height/width
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

            crop_images.append(
                F.upsample_bilinear(
                    images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                    size=(imgH, imgW)))
        crop_images = torch.cat(crop_images, dim=0)
        return crop_images

    elif mode == 'drop':
        drop_masks = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                if percent_max:
                    theta_d = random.uniform(*theta) * atten_map.max()
                else:
                    theta_c = random.uniform(*theta) * atten_map.mean()
            else:
                if percent_max:
                    theta_c = theta * atten_map.max()
                else:
                    theta_c = theta * atten_map.mean()

            drop_masks.append(F.upsample_bilinear(atten_map, size=(imgH, imgW)) < theta_d)
        drop_masks = torch.cat(drop_masks, dim=0)
        drop_images = images * drop_masks.float()
        return drop_images

    else:
        raise ValueError('Expected mode in [\'crop\', \'drop\'], \
            but received unsupported augmentation method %s' % mode)


class BasicConv2D(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


# Bilinear Attention Pooling
class BAP_Counterfactual(nn.Module):
    def __init__(self, pool='GAP'):
        super(BAP_Counterfactual, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pool is None:
            feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)

        # sign-sqrt
        feature_matrix_raw = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + EPSILON)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix_raw, dim=-1)

        if self.training:
            fake_att = torch.zeros_like(attentions).uniform_(0, 2)
        else:
            fake_att = torch.ones_like(attentions)
        counterfactual_feature = (torch.einsum('imjk,injk->imn', (fake_att, features)) / float(H * W)).view(B, -1)

        counterfactual_feature = torch.sign(counterfactual_feature) * torch.sqrt(
            torch.abs(counterfactual_feature) + EPSILON)

        counterfactual_feature = F.normalize(counterfactual_feature, dim=-1)
        return feature_matrix, counterfactual_feature


class WSDAN_CAL(nn.Module):
    """
    WS-DAN models
    Hu et al.,
    "See Better Before Looking Closer: Weakly Supervised Data Augmentation Network
    for Fine-Grained Visual Classification",
    arXiv:1901.09891
    """
    def __init__(self, num_classes, num_features=2048, num_attention_maps=32,
                 num_aug_maps=None, cam_aug=False):
        super(WSDAN_CAL, self).__init__()
        # Attention Maps
        self.num_attention_maps = num_attention_maps
        self.num_aug_maps = num_aug_maps
        self.cam_aug = cam_aug

        if cam_aug:
            self.cam_head = nn.Linear(num_features, num_classes, bias=False)

        else:
            self.attentions = BasicConv2D(num_features, num_attention_maps, kernel_size=1)

            # Bilinear Attention Pooling
            self.bap = BAP_Counterfactual(pool='GAP')

            # Classification Layer
            self.fc = nn.Linear(num_attention_maps * num_features, num_classes, bias=False)

    def forward(self, feature_maps):
        batch_size = feature_maps.size(0)

        if hasattr(self, 'cam_head'):
            feature_matrix = reduce(feature_maps, 'b c h w -> b c', 'mean')
            logits = self.cam_head(feature_matrix)

            feature_maps_reshaped = rearrange(feature_maps, 'b c h w -> b h w c')
            cams_all = self.cam_head(feature_maps_reshaped)
            cams_all = rearrange(cams_all, 'b h w k -> b k h w')

            top_pred = logits.argmax(dim=-1)
            cams = torch.stack([cams_all[i, top_pred[i]] for i in range(batch_size)])
            # original paper does not normalize
            # Learning Deep Features for Discriminative Localization
            # cams = F.relu(cams)
            # cams_min = reduce(cams, 'b h w -> b 1 1', 'min')
            # cams_max = reduce(cams, 'b h w -> b 1 1', 'max')
            # cams = (cams - cams_min) / (cams_max - cams_min)
            cams = repeat(cams, 'b h w -> b n h w', n=2)

            return logits, logits, feature_matrix, cams, cams_all

        # Feature Maps, Attention Maps and Feature Matrix
        attention_maps = self.attentions(feature_maps)

        feature_matrix, feature_matrix_hat = self.bap(feature_maps, attention_maps)

        # Classification
        p = self.fc(feature_matrix * 100.)

        # Generate Attention Map
        if self.training:
            # Randomly choose one of attention maps Ak
            attention_map = []

            # restrict number of maps for augmentation during student training
            if self.num_aug_maps:
                aug_maps = attention_maps[:, :self.num_aug_maps]
                for i in range(batch_size):
                    attention_weights = torch.sqrt(aug_maps[i].sum(dim=(1, 2)).detach() + EPSILON)
                    attention_weights = F.normalize(attention_weights, p=1, dim=0)
                    k_index = np.random.choice(aug_maps.shape[1], 2, p=attention_weights.cpu().numpy())
                    attention_map.append(aug_maps[i, k_index, ...])

            else:
                for i in range(batch_size):
                    attention_weights = torch.sqrt(attention_maps[i].sum(dim=(1, 2)).detach() + EPSILON)
                    attention_weights = F.normalize(attention_weights, p=1, dim=0)
                    k_index = np.random.choice(self.num_attention_maps, 2, p=attention_weights.cpu().numpy())
                    attention_map.append(attention_maps[i, k_index, ...])

            attention_map = torch.stack(attention_map)  # (B, 2, H, W) - one for cropping, the other for dropping

        else:
            attention_map = torch.mean(attention_maps, dim=1, keepdim=True)  # (B, 1, H, W)

        return p, p - self.fc(feature_matrix_hat * 100.), feature_matrix, attention_map, attention_maps


class ContrastiveMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ContrastiveMLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size),
            nn.BatchNorm1d(output_size, affine=False),
        )

    def forward(self, x):
        # B, D -> B, D*ER -> B, D
        return self.mlp(x)


class DiscriminativeAwarePooling(nn.Module):
    def __init__(self, pooling_function='gap', agap_kernel=5,
                 layout='bsd', width=None, sign_sqrt=False, normalization=False):
        super(DiscriminativeAwarePooling, self).__init__()

        self.pooling_function = pooling_function
        if pooling_function == 'gap' and layout == 'bsd':
            self.reducer = Reduce('b s d -> b d', 'mean')
        elif pooling_function == 'gap' and layout == 'bchw':
            self.reducer = Reduce('b c h w -> b c', 'mean')
        elif pooling_function == 'gap' and layout == 'bhwc':
            self.reducer = Reduce('b h w c -> b c', 'mean')
        elif pooling_function in ('abap', 'bap', 'aadgmp', 'adgmp', 'aadgap', 'adgap', 'agap_mean', 'agap_random') and layout == 'bsd':
            self.reducer = Rearrange('b (h w) d -> b d h w', w=width)
        elif pooling_function in ('abap', 'bap', 'aadgmp', 'adgmp', 'aadgap', 'adgap', 'agap_mean', 'agap_random') and layout == 'bchw':
            self.reducer = nn.Identity()
        elif pooling_function in ('abap', 'bap', 'aadgmp', 'adgmp', 'aadgap', 'adgap', 'agap_mean', 'agap_random') and layout == 'bhwc':
            self.reducer = Rearrange('b h w c -> b c h w')
        else:
            raise NotImplementedError

        if pooling_function in ('aadgmp', 'adgmp'):
            self.gmp = Reduce('b c h w -> b c', 'max')
        elif pooling_function in ('aadgap', 'adgap'):
            self.gap = Reduce('b c h w -> b c', 'mean')
        elif pooling_function in ('agap_mean', 'agap_random'):
            self.agap = nn.AvgPool2d(agap_kernel, stride=1)

        self.sign_sqrt = sign_sqrt
        self.normalization = normalization

    def forward(self, features, attentions):
        features = self.reducer(features)

        if self.pooling_function in ('abap', 'bap', 'aadgmp', 'adgmp', 'aadgap', 'adgap', 'agap_mean', 'agap_random'):
            b, c, h, w = features.shape
            _, m, ah, aw = attentions.size()

            # match size
            if ah != h or aw != w:
                attentions = F.upsample_bilinear(attentions, size=(h, w))

            # bilinear attention pooling: B, M, (HxW) x B, (HxW), C -> B, M, C
            # feature_matrix: (B, M, C) -> (B, M * C)
            if self.pooling_function == 'abap':
                attentions = reduce(attentions, 'b ac ah aw -> b 1 ah aw', 'mean')
                m = attentions.shape[1]

            if self.pooling_function in ('abap', 'bap'):
                feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(h * w)).view(b, -1)

            # attention driven global max pooling -> B, C, H, W -> times attention -> B, C, H, W ->
            # max pool -> B, C -> B, C -> repeat for all M -> B, C * M
            elif self.pooling_function in ('aadgmp', 'adgmp', 'aadgap', 'adgap'):
                feature_matrix = []

                if self.pooling_function in ('aadgmp', 'aadgap'):
                    attentions = reduce(attentions, 'b ac ah aw -> b 1 ah aw', 'mean')
                    m = attentions.shape[1]

                for i in range(m):
                    if self.pooling_function in ('aadgmp', 'adgmp'):
                        AiF = self.gmp(features * attentions[:, i:i + 1, ...])
                    elif self.pooling_function in ('aadgap', 'adgap'):
                        AiF = self.gap(features * attentions[:, i:i + 1, ...])
                    else:
                        raise NotImplementedError
                    feature_matrix.append(AiF)
                feature_matrix = torch.cat(feature_matrix, dim=1)

            # attention guided average pooling -> B, C, H, W -> pool -> B, C, PH, PW ->
            # flatten into 1d -> B, C, (PHxPW), similarly for attention
            # select index of top-1 pooled attention as region with highest importance
            # use index to gather corresponding average pooled of features (GAP of discriminative region)
            elif 'agap' in self.pooling_function:
                features = self.agap(features)
                attentions = self.agap(attentions)

                features_1d = rearrange(features, 'b c h w -> b (h w) c')
                attentions_1d = rearrange(attentions, 'b c h w -> b (h w) c')

                if self.pooling_function == 'agap_mean':
                    attentions_1d = reduce(attentions_1d, 'b s c -> b s', 'mean')
                elif self.pooling_function == 'agap_random':
                    attention = []
                    for i in range(b):
                        attention_weights = torch.sqrt(attentions_1d[i].sum(dim=0).detach() + EPSILON)
                        attention_weights = F.normalize(attention_weights, p=1, dim=0)
                        k_index = np.random.choice(m, 1, p=attention_weights.cpu().numpy())
                        attention.append(attentions_1d[i, :, k_index])
                    attention = torch.stack(attention, dim=0)  # (B, S, 1)
                    attentions_1d = rearrange(attention, 'b s 1 -> b s')

                _, top_idx = attentions_1d.topk(1, dim=-1, largest=True)
                top_idx = repeat(top_idx, 'b 1 -> b 1 c', c=c)

                feature_matrix = torch.gather(features_1d, 1,top_idx)
                feature_matrix = rearrange(feature_matrix, 'b 1 c -> b c')
        else:
            feature_matrix = features

        if self.sign_sqrt:
            # sign-sqrt for numerical stability
            feature_matrix = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + EPSILON)

        if self.normalization:
            # l2 normalization along dimension M and C
            feature_matrix = F.normalize(feature_matrix, p=2, dim=-1)

        return feature_matrix


class CAL(nn.Module):
    def __init__(
        self,
        model,
        seq_len=196,
        output_size=768,
        num_classes=1000,
        shape_format='bsd',
        device='cpu',
        teacher=False,
        student=False,
        tgda=False,
        kd_aux_loss=None, 
        num_images=10000,
        cont_negatives=4096,
        cont_temp=0.7,
        cont_loss=False,
        image_size=224,
        layers=None,
        pooling_function=None,
        pool_size=None,
        disc_feats_norm=False,
        disc_feats_sign_sqrt=False,
        if_channels=[],
        mlp_ratio=4,
        pre_resize_factor=None,
        student_image_size=None,
        cal_ap_only=False,
        th_crop=0.5,
        th_mask=0.35,
        beta=5e-2,
        num_attention_maps=32,
        num_aug_maps=None,
        cam_aug=False,
        train_both=False,
        teacher_loss=None,
    ):
        super(CAL, self).__init__()

        self.image_size = image_size
        self.teacher = teacher
        self.student = student
        self.tgda = tgda
        self.beta = beta
        self.th_crop = th_crop
        self.th_mask = th_mask

        self.cal_ap_only = cal_ap_only

        if train_both and teacher_loss == 'cal':
            self.train_teacher_cal = True

        if pre_resize_factor:
            self.pre_resize_factor = pre_resize_factor
        if student_image_size:
            self.student_image_size = student_image_size

        # Network Initialization
        if shape_format == 'bsd':
            ph = int(math.sqrt(seq_len))
            self.encoder = nn.Sequential(
                model,
                Rearrange('b (h w) d -> b d h w', h=ph)
            )
        elif shape_format == 'bhwd':
            self.encoder = nn.Sequential(
                model,
                Rearrange('b h w d -> b d h w')
            )
        else:
            self.encoder = model

        if student:
            self.dfsm = nn.Sequential(
                Reduce('b d h w -> b d', 'mean'),
                nn.Linear(output_size, num_classes)
            )
        else:
            # discriminative feature selection mechanism
            self.dfsm = WSDAN_CAL(num_classes, output_size, num_attention_maps,
                                  num_aug_maps, cam_aug)

            self.feature_center = torch.zeros(
                num_classes, num_attention_maps * output_size, device=device)

            print('WSDAN: num_attention_maps: {}'.format(num_attention_maps),
                'Threshhold crop and mask: {}, {}'.format(th_crop, th_mask))

        if (cont_loss or kd_aux_loss == 'crd') and teacher:
            self.layers = []
            self.if_dic = {}

            self._insert_hooks(layers)
            print('Contrastive layers from teacher: ', self.layers)

            widths, self.if_channels, layouts = self.get_if_dims(image_size)

            if pooling_function == 'bapto':
                pooling_function = 'bap'
                print('BAPTO behaves as BAP for Teacher')

            self.if_pools = nn.ModuleList([
                DiscriminativeAwarePooling(
                    pooling_function, pool_size, layout, width, disc_feats_sign_sqrt, disc_feats_norm,
                ) for width, layout in zip(widths, layouts)
            ])

        elif (cont_loss or kd_aux_loss == 'crd') and student:
            if layers is not None:
                self.layers = []
                self.if_dic = {}

                self._insert_hooks(layers)
                print('Contrastive layers from student: ', self.layers)

                widths, if_channels_student, layouts = self.get_if_dims(image_size)
            else:                
                widths = [None]
                if_channels_student = [output_size]
                layouts = ['bchw']

            # adjust intermediate feature number of channels for contrastive MLP
            if pooling_function == 'bapto':
                if_channels = [c * num_attention_maps for c in if_channels]

            if_channels = if_channels + if_channels_student

            if pooling_function in ('bap', 'adgmp', 'adgap'):
                if_channels = [c * num_attention_maps for c in if_channels]

            # bapto behaves as gap for student
            if pooling_function == 'bapto':
                pooling_function = 'gap'
                print('BAPTO behaves as GAP for Student')

            self.if_pools = nn.ModuleList([
                DiscriminativeAwarePooling(
                    pooling_function, pool_size, layout, width, disc_feats_sign_sqrt, disc_feats_norm,
                ) for width, layout in zip(widths, layouts)
            ])

            self.cont_mlp = nn.ModuleList([
                 ContrastiveMLP(c, int(output_size * mlp_ratio), output_size) for c in if_channels
            ])

            if kd_aux_loss == 'crd':
                self.criterion_crd = CRDLoss(num_images, output_size, cont_negatives, cont_temp)

    def _log_layer(self,
                   name: str,
                   layer: nn.Module,
                   inp: torch.Tensor,
                   out: torch.Tensor):

        self.if_dic[name] = out

    def _insert_hooks(self, layers):
        for name, layer in self.named_modules():
            if layers is not None:
                if name in layers:
                    self.layers += [name]
                    layer.register_forward_hook(partial(self._log_layer, name))
            else:
                self.layers += [name]
                layer.register_forward_hook(partial(self._log_layer, name))

    def get_if_dims(self, image_size):
        self.if_dic = {}

        with torch.no_grad():
            x = torch.rand(2, 3, image_size, image_size)
            _ = self.encoder(x)

            # -1 in certain cases corresponds to classification layer
            features = list(self.if_dic.values())

            widths = []
            channels = []
            layouts = []

            for ft in features:
                if len(ft.shape) == 4:
                    b, c, h, w = ft.shape

                    if h != w:
                        # b, h, w, c is the actual shape
                        temp1 = h
                        temp2 = w
                        temp3 = d
                        h = temp1
                        w = temp2
                        d = temp3
                        layouts.append('bhwc')
                    else:
                        layouts.append('bchw')

                elif len(ft.shape) == 3:
                    b, s, c = ft.shape
                    w = int(s ** 0.5)
                    layouts.append('bsd')
                else:
                    raise NotImplementedError

                widths.append(w)
                channels.append(c)
        return widths, channels, layouts

    def forward(self, images, y=None, output_t=None, idx=None, sample_idx=None):
        if hasattr(self, 'pre_resize_factor'):
            with torch.no_grad():
                x = F.interpolate(images, size=(self.image_size, self.image_size), mode='bicubic')
        else:
            x = images 

        self.if_dic = {}


        if self.training and self.student and self.tgda and hasattr(self, 'cont_mlp') and output_t is not None:
            if hasattr(self, 'student_image_size'):
                with torch.no_grad():
                    x = F.interpolate(x, size=(self.student_image_size, self.student_image_size), mode='bicubic')

            # raw image
            feature_maps = self.encoder(x)
            y_pred_raw = self.dfsm(feature_maps)

            _, _, aug_images, feats_t, feats_t_aug, attention_maps, attention_maps_aug = output_t

            # feats_s = [self.if_pool(feature_maps, attention_maps)]
            # feats_s = [feature_maps]
            feats_s = list(self.if_dic.values())
            # print('student: ', self.if_dic.keys(), [ft.shape for ft in feats_s])
            feats_s = [pool(ft, attention_maps) for pool, ft in zip(self.if_pools, feats_s)]
            # print('student post pool: ', [ft.shape for ft in feats_s])
            self.if_dic = {}

            if hasattr(self, 'student_image_size'):
                with torch.no_grad():
                    aug_images = F.interpolate(aug_images, size=(self.student_image_size, self.student_image_size), mode='bicubic')

            # crop images forward
            feature_maps_aug = self.encoder(aug_images)
            y_pred_aug = self.dfsm(feature_maps_aug)

            # feats_s_aug = [self.if_pool(feature_maps_aug, attention_maps_aug)]
            # feats_s_aug = [feature_maps_aug]
            feats_s_aug = list(self.if_dic.values())
            feats_s_aug = [pool(ft, attention_maps_aug) for pool, ft in zip(self.if_pools, feats_s_aug)]

            # for student: batch_size x channels_out
            # for teacher: batch_size x channels_out, for X layers
            feats = feats_t + feats_s
            feats_aug = feats_t_aug + feats_s_aug

            feats = [mlp(ft) for mlp, ft in zip(self.cont_mlp, feats)]
            feats_aug = [mlp(ft) for mlp, ft in zip(self.cont_mlp, feats_aug)]

            feats = torch.stack(feats, dim=1)
            feats_aug = torch.stack(feats_aug, dim=1)

            feats_aug_crops, feats_aug_drops = torch.split(feats_aug, [x.shape[0], x.shape[0]], dim=0)
            # feats = torch.cat([feats, feats_aug_crops, feats_aug_drops], dim=1)
            feats = torch.cat([feats, feats_aug_crops], dim=1)
            # print(feats.shape)

            return (y_pred_raw, y_pred_aug, feats, aug_images)


        elif self.training and self.student and self.tgda and output_t is not None:
            if hasattr(self, 'student_image_size'):
                with torch.no_grad():
                    x = F.interpolate(x, size=(self.student_image_size, self.student_image_size), mode='bicubic')

            # raw image
            feature_maps = self.encoder(x)
            y_pred_raw = self.dfsm(feature_maps)

            aug_images = output_t[-1]

            if hasattr(self, 'student_image_size'):
                with torch.no_grad():
                    aug_images = F.interpolate(aug_images, size=(self.student_image_size, self.student_image_size), mode='bicubic')

            # crop images forward
            feature_maps = self.encoder(aug_images)
            y_pred_aug = self.dfsm(feature_maps)

            return (y_pred_raw, y_pred_aug, aug_images)


        elif self.training and self.student and hasattr(self, 'cont_mlp') and output_t is not None:
            if hasattr(self, 'student_image_size'):
                with torch.no_grad():
                    x = F.interpolate(x, size=(self.student_image_size, self.student_image_size), mode='bicubic')

            # raw image
            feature_maps = self.encoder(x)
            y_pred_raw = self.dfsm(feature_maps)

            _, feats_t, attention_maps = output_t

            # feats_s = [self.if_pool(feature_maps, attention_maps)]
            feats_s = [feature_maps]
            # feats_s = list(self.if_dic.values())
            feats_s = [pool(ft, attention_maps) for pool, ft in zip(self.if_pools, feats_s)]

            feats = feats_t + feats_s
            feats = [mlp(ft) for mlp, ft in zip(self.cont_mlp, feats)]

            if hasattr(self, 'criterion_crd'):
                feats_t, feats_s = feats[-2], feats[-1]
                loss_crd = self.criterion_crd(feats_s, feats_t, idx, sample_idx)
                return y_pred_raw, feats, loss_crd
    
            feats = torch.stack(feats, dim=1)
            return y_pred_raw, feats


        elif self.teacher and hasattr(self, 'if_pools') and self.tgda:
            # raw image
            feature_maps = self.encoder(x)
            y_pred_raw, _, _, attention_map, attention_maps = self.dfsm(feature_maps)

            feats = list(self.if_dic.values())
            # print('teacher: ', self.if_dic.keys(), [ft.shape for ft in feats])
            feats = [pool(ft, attention_maps) for pool, ft in zip(self.if_pools, feats)]
            self.if_dic = {}

            # attention cropping
            with torch.no_grad():
                if not self.training:
                    attention_map = torch.cat([attention_map, attention_map], dim=1)

                crop_images = batch_augment(
                    images, attention_map[:, :1, :, :], mode='crop',
                    theta=(self.th_crop-0.1, self.th_crop+0.1), padding_ratio=0.1)
                drop_images = batch_augment(images, attention_map[:, 1:, :, :], mode='drop',
                                            theta=(self.th_mask-0.15, self.th_mask+0.15))
            aug_images = torch.cat([crop_images, drop_images], dim=0)

            # crop images forward
            feature_maps = self.encoder(aug_images)
            y_pred_aug, _, _, _, attention_maps_aug = self.dfsm(feature_maps)

            feats_aug = list(self.if_dic.values())
            feats_aug = [pool(ft, attention_maps_aug) for pool, ft in zip(self.if_pools, feats_aug)]

            return (y_pred_raw, y_pred_aug, aug_images, feats, feats_aug, attention_maps, attention_maps_aug)


        elif self.student:
            if hasattr(self, 'student_image_size'):
                with torch.no_grad():
                    x = F.interpolate(x, size=(self.student_image_size, self.student_image_size), mode='bicubic')
            feature_maps = self.encoder(x)
            y_pred = self.dfsm(feature_maps)
            return y_pred


        elif hasattr(self, 'train_teacher_cal') and self.teacher and self.tgda:
            # raw image
            feature_maps = self.encoder(x)
            y_pred_raw, y_pred_aux, feature_matrix, attention_map, attention_maps = self.dfsm(feature_maps)

            # Update Feature Center
            feature_center_batch = F.normalize(self.feature_center[y], dim=-1)
            self.feature_center[y] += self.beta * (feature_matrix.detach() - feature_center_batch)

            # attention cropping
            with torch.no_grad():
                crop_images = batch_augment(
                    images, attention_map[:, :1, :, :], mode='crop',
                    theta=(self.th_crop-0.1, self.th_crop+0.1), padding_ratio=0.1)
                drop_images = batch_augment(images, attention_map[:, 1:, :, :], mode='drop',
                                            theta=(self.th_mask-0.15, self.th_mask+0.15))
            aug_images = torch.cat([crop_images, drop_images], dim=0)

            # crop images forward
            feature_maps = self.encoder(aug_images)
            y_pred_aug, y_pred_aux_aug, _, _, attention_maps = self.dfsm(feature_maps)

            y_pred_aux = torch.cat([y_pred_aux, y_pred_aux_aug], dim=0)

            # final prediction
            y_pred_aug_crops, _ = torch.split(y_pred_aug, [x.shape[0], x.shape[0]], dim=0)
            y_pred = (y_pred_raw + y_pred_aug_crops) / 2.

            return (y_pred, y_pred_raw, y_pred_aux, feature_matrix, feature_center_batch,
                    y_pred_aug, aug_images)


        elif self.teacher and self.tgda:
            # raw image
            feature_maps = self.encoder(x)
            y_pred_raw, _, _, attention_map, attention_maps = self.dfsm(feature_maps)

            # attention cropping
            with torch.no_grad():
                if not self.training:
                    attention_map = torch.cat([attention_map, attention_map], dim=1)

                crop_images = batch_augment(
                    images, attention_map[:, :1, :, :], mode='crop',
                    theta=(self.th_crop-0.1, self.th_crop+0.1), padding_ratio=0.1)
                drop_images = batch_augment(images, attention_map[:, 1:, :, :], mode='drop',
                                            theta=(self.th_mask-0.15, self.th_mask+0.15))
            aug_images = torch.cat([crop_images, drop_images], dim=0)

            # crop images forward
            feature_maps = self.encoder(aug_images)
            y_pred_aug, _, _, _, attention_maps = self.dfsm(feature_maps)

            return (y_pred_raw, y_pred_aug, aug_images)


        elif self.teacher:
            feature_maps = self.encoder(x)
            y_pred, _, _, _, attention_maps = self.dfsm(feature_maps)

            return y_pred


        elif self.training and y is not None:
            # raw image
            feature_maps = self.encoder(x)
            y_pred_raw, y_pred_aux, feature_matrix, attention_map, attention_maps = self.dfsm(feature_maps)

            # Update Feature Center
            feature_center_batch = F.normalize(self.feature_center[y], dim=-1)
            self.feature_center[y] += self.beta * (feature_matrix.detach() - feature_center_batch)

            # attention cropping
            with torch.no_grad():
                crop_images = batch_augment(
                    images, attention_map[:, :1, :, :], mode='crop',
                    theta=(self.th_crop-0.1, self.th_crop+0.1), padding_ratio=0.1)
                drop_images = batch_augment(images, attention_map[:, 1:, :, :], mode='drop',
                                            theta=(self.th_mask-0.15, self.th_mask+0.15))
            aug_images = torch.cat([crop_images, drop_images], dim=0)

            # crop images forward
            feature_maps = self.encoder(aug_images)
            y_pred_aug, y_pred_aux_aug, _, _, attention_maps = self.dfsm(feature_maps)

            y_pred_aux = torch.cat([y_pred_aux, y_pred_aux_aug], dim=0)

            # final prediction
            y_pred_aug_crops, _ = torch.split(y_pred_aug, [x.shape[0], x.shape[0]], dim=0)
            y_pred = (y_pred_raw + y_pred_aug_crops) / 2.

            return (y_pred, y_pred_raw, y_pred_aux, feature_matrix, feature_center_batch,
                    y_pred_aug, aug_images)


        elif self.cal_ap_only:
            # Raw Image
            feature_maps = self.encoder(x)
            y_pred_raw, _, _, _, _ = self.dfsm(feature_maps)
            return y_pred_raw


        else:
            if hasattr(self, 'pre_resize_factor'):
                with torch.no_grad():
                    images_m = torch.flip(images, [3])
            else:
                images_m = images
            x_m = torch.flip(x, [3])

            # Raw Image
            feature_maps = self.encoder(x)
            y_pred_raw, _, _, attention_map, _ = self.dfsm(feature_maps)

            feature_maps = self.encoder(x_m)
            y_pred_raw_m, _, _, attention_map_m, _ = self.dfsm(feature_maps)

            # Object Localization and Refinement
            crop_images = batch_augment(images, attention_map, mode='crop', theta=0.3, padding_ratio=0.1)
            feature_maps = self.encoder(crop_images)
            y_pred_crop, _, _, _, _ = self.dfsm(feature_maps)

            crop_images2 = batch_augment(images, attention_map, mode='crop', theta=0.2, padding_ratio=0.1)
            feature_maps = self.encoder(crop_images2)
            y_pred_crop2, _, _, _, _ = self.dfsm(feature_maps)

            crop_images3 = batch_augment(images, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
            feature_maps = self.encoder(crop_images3)
            y_pred_crop3, _, _, _, _ = self.dfsm(feature_maps)

            crop_images_m = batch_augment(images_m, attention_map_m, mode='crop', theta=0.3, padding_ratio=0.1)
            feature_maps = self.encoder(crop_images_m)
            y_pred_crop_m, _, _, _, _ = self.dfsm(feature_maps)

            crop_images_m2 = batch_augment(images_m, attention_map_m, mode='crop', theta=0.2, padding_ratio=0.1)
            feature_maps = self.encoder(crop_images_m2)
            y_pred_crop_m2, _, _, _, _ = self.dfsm(feature_maps)

            crop_images_m3 = batch_augment(images_m, attention_map_m, mode='crop', theta=0.1, padding_ratio=0.05)
            feature_maps = self.encoder(crop_images_m3)
            y_pred_crop_m3, _, _, _, _ = self.dfsm(feature_maps)

            y_pred = (y_pred_raw + y_pred_crop + y_pred_crop2 + y_pred_crop3) / 4.
            y_pred_m = (y_pred_raw_m + y_pred_crop_m + y_pred_crop_m2 + y_pred_crop_m3) / 4.
            y_pred = (y_pred + y_pred_m) / 2.

            # return y_pred
            return y_pred, crop_images
