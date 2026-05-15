import os
import random
import argparse
import json
from warnings import warn
from typing import List, Dict
from pathlib import Path
from functools import partial
from contextlib import suppress
from statistics import mean, stdev

import numpy as np
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
from einops import rearrange, reduce
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from fgir_kd.data_utils.build_dataloaders import build_dataloaders
from fgir_kd.other_utils.build_args import parse_inference_args
from fgir_kd.model_utils.build_model import build_model
from fgir_kd.train_utils.misc_utils import set_random_seed


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams.update({'font.size': 15})


MODELS_DIC = {
    'vit_b16': 'ViT',
}

DATASETS_DIC = {
    'aircraft': 'Aircraft',
    'cars': 'Cars',
    'cotton': 'Cotton',
    'cub': 'CUB',
    'dafb': 'DAFB',
    'dogs': 'Dogs',
    'flowers': 'Flowers',
    'food': 'Food',
    'inat17': 'iNat17',
    'moe': 'Moe',
    'nabirds': 'NABirds',
    'pets': 'Pets',
    'soyageing': 'SoyAgeing',
    'soyageingr1': 'SoyAgeingR1',
    'soyageingr3': 'SoyAgeingR3',
    'soyageingr4': 'SoyAgeingR4',
    'soyageingr5': 'SoyAgeingR5',
    'soyageingr6': 'SoyAgeingR6',
    'soygene': 'SoyGene',
    'soyglobal': 'SoyGlobal',
    'soylocal': 'SoyLocal',
    'vegfru': 'VegFru',
}

SETTINGS_DIC = {
    'scratch': '(Randomly Initialized)',
    'fz': '(Frozen without PETL Modules)',
    'ft': '(Fine-Tuned)',
    'pt': 'Pre-trained',
    'cal': 'CAL'
}


def adjust_args_general(args):
    selector = f'_{args.selector}' if args.selector else ''

    args.run_name = '{}_{}_{}{}_{}_{}'.format(
        args.dataset_name, args.model_name, args.model_name_teacher, selector, args.setting, args.serial
    )

    args.results_dir = os.path.join(args.results_inference, args.run_name)
    os.makedirs(args.results_dir, exist_ok=True)
    return args


def add_colorbar(im, aspect=10, pad_fraction=0.2, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

class FeatureMetrics:
    def __init__(self,
                 model: nn.Module,
                 model2: nn.Module,
                 model_name: str = None,
                 model_layers: List[str] = None,
                 model2_name: str = None,
                 model2_layers: List[str] = None,
                 device: str ='cpu',
                 image_size: int = 224,
                 setting: str = 'fz',
                 out_size: int = 7,
                 debugging: bool = False):
        """

        :param model: (nn.Module) Neural Network 1
        :param model_name: (str) Name of model 1
        :param model_layers: (List) List of layers to extract features from
        :param device: Device to run the model
        """

        self.model = model
        self.model2 = model2

        self.device = device

        # model 1
        self.model_info = {}

        self.model_info['Setting'] = SETTINGS_DIC.get(setting, setting)

        if model_name is None:
            self.model_info['Name_og'] = model.__repr__().split('(')[0]
        else:
            self.model_info['Name_og'] = model_name
        self.model_info['Name'] = MODELS_DIC.get(self.model_info['Name_og'], self.model_info['Name_og'])

        self.model_info['Layers'] = []

        self.model_features = {}

        if len(list(model.modules())) > 150 and model_layers is None:
            warn("Model 1 seems to have a lot of layers. " \
                 "Consider giving a list of layers whose features you are concerned with " \
                 "through the 'model_layers' parameter. Your CPU/GPU will thank you :)")

        self.model_layers = model_layers


        # model2
        self.model2_info = {}

        self.model2_info['Setting'] = SETTINGS_DIC.get(setting, setting)

        if model2_name is None:
            self.model2_info['Name_og'] = model2.__repr__().split('(')[0]
        else:
            self.model2_info['Name_og'] = model2_name
        self.model2_info['Name'] = MODELS_DIC.get(self.model2_info['Name_og'], self.model2_info['Name_og'])

        self.model2_info['Layers'] = []

        self.model2_features = {}

        if len(list(model2.modules())) > 150 and model2_layers is None:
            warn("Model 2 seems to have a lot of layers. " \
                 "Consider giving a list of layers whose features you are concerned with " \
                 "through the 'model2_layers' parameter. Your CPU/GPU will thank you :)")

        self.model2_layers = model2_layers


        self._insert_hooks()
        self.model = self.model.to(self.device)
        self.model2 = self.model2.to(self.device)

        self.model.eval()
        self.model2.eval()

        self._check_shape(image_size)

        self.pool = nn.AdaptiveAvgPool2d((out_size, out_size)).to(self.device)

        self.debugging = debugging

        print(self.model_info)
        print(self.model2_info)

    def _log_layer(self,
                   model: str,
                   name: str,
                   layer: nn.Module,
                   inp: torch.Tensor,
                   out: torch.Tensor):

        if model == "model":
            self.model_features[name] = out
        elif model == "model2":
            self.model2_features[name] = out
        else:
            raise RuntimeError("Unknown model name for _log_layer.")

    def _insert_hooks(self):
        # Model 1
        for name, layer in self.model.named_modules():
            if self.model_layers is not None:
                if name in self.model_layers:
                    self.model_info['Layers'] += [name]
                    layer.register_forward_hook(partial(self._log_layer, "model", name))
            else:
                self.model_info['Layers'] += [name]
                layer.register_forward_hook(partial(self._log_layer, "model", name))

        # Model 2
        for name, layer in self.model2.named_modules():
            if self.model2_layers is not None:
                if name in self.model2_layers:
                    self.model2_info['Layers'] += [name]
                    layer.register_forward_hook(partial(self._log_layer, "model2", name))
            else:
                self.model2_info['Layers'] += [name]
                layer.register_forward_hook(partial(self._log_layer, "model2", name))

    def _check_shape(self, image_size):
        with torch.no_grad():
            x = torch.rand(2, 3, image_size, image_size).to(self.device)
            _ = self.model(x)

            # -1 in certain cases corresponds to classification layer
            last = self.model_info['Layers'][-2]
            feat_out = self.model_features[last]

            if len(feat_out.shape) == 4:
                b, c, h, w = feat_out.shape
                if h == w:
                    self.bhcw_model = True
                    h = feat_out.shape[-1]
                else:
                    self.bhcw_model = False
                    h = feat_out.shape[1]
            elif len(feat_out.shape) == 3:
                h = int(feat_out.shape[1] ** 0.5)
                self.cls_model = False if h ** 2 == feat_out.shape[1] else True
            else:
                pass


            # model2
            _ = self.model2(x)

            # -1 in certain cases corresponds to classification layer
            last = self.model2_info['Layers'][-2]
            feat_out = self.model2_features[last]

            if len(feat_out.shape) == 4:
                b, c, h, w = feat_out.shape
                if h == w:
                    self.bhcw_model2 = True
                    h = feat_out.shape[-1]
                else:
                    self.bhcw_model2 = False
                    h = feat_out.shape[1]
            elif len(feat_out.shape) == 3:
                h = int(feat_out.shape[1] ** 0.5)
                self.cls_model2 = False if h ** 2 == feat_out.shape[1] else True
            else:
                pass


    def _HSIC(self, K, L):
        """
        Computes the unbiased estimate of HSIC metric.

        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N = K.shape[0]
        ones = torch.ones(N, 1).to(self.device)
        result = torch.trace(K @ L)
        result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
        result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
        return (1 / (N * (N - 3)) * result).item()

    def _pool_features(self, feat, pool=True, model2=False):
        if hasattr(self, 'pool') and pool:
            if len(feat.shape) == 4:
                if (model2 and not self.bhcw_model2) or (not self.bhcw_model):
                    feat = rearrange(feat, 'b h w c -> b c h w')
                pooled = self.pool(feat)
                pooled = rearrange(pooled, 'b c h w -> (b h w) c')
            elif len(feat.shape) == 3:
                h = int(feat.shape[1] ** 0.5)
                if (model2 and self.cls_model) or self.cls_model:
                    x_cls, x_others = torch.split(feat, [1, int(h**2)], dim=1)
                    x_others = rearrange(x_others, 'b (h w) d -> b d h w', h=h)
                    x_others = self.pool(x_others)
                    x_others = rearrange(x_others, 'b d h w -> b (h w) d')
                    pooled = torch.cat([x_cls, x_others], dim=1)
                    pooled = rearrange(pooled, 'b s d -> (b s) d')
                else:
                    pooled = rearrange(feat, 'b (h w) d -> b d h w', h=h)
                    pooled = self.pool(pooled)
                    pooled = rearrange(pooled, 'b c h w -> (b h w) c')                    
        else:
            pooled = feat.flatten(1)
 
        return pooled

    def compare(self,
                dataloader1: DataLoader) -> None:
        """
        Computes the feature similarity between the models on the
        given datasets.
        :param dataloader1: (DataLoader)
        """

        self.model_info['Dataset_og'] = dataloader1.dataset.dataset_name
        self.model_info['Dataset'] = DATASETS_DIC.get(self.model_info['Dataset_og'], self.model_info['Dataset_og'])

        layers = self.model_layers if self.model_layers is not None else list(self.model.modules())
        layers2 = self.model2_layers if self.model2_layers is not None else list(self.model2.modules())

        N = len(layers)
        M = len(layers2)
        self.hsic_matrix = torch.zeros(N, M, 3)
        self.dist_cum = torch.zeros(N, device=self.device)
        self.dist_cum_norm = torch.zeros(N, device=self.device)
        self.l2_norm = torch.zeros(N, device=self.device)

        num_batches = len(dataloader1)

        for (x1, *_) in tqdm(dataloader1, desc="| Comparing features |", total=num_batches):
            x1 = x1.to(self.device)

            self.model_features = {}
            _ = self.model(x1)

            self.model2_features = {}
            _ = self.model2(x1)

            self.compare_cka_l2_dist(num_batches)

        self.hsic_matrix = self.hsic_matrix[:, :, 1] / (self.hsic_matrix[:, :, 0].sqrt() *
                                                        self.hsic_matrix[:, :, 2].sqrt())

    def compare_cka_l2_dist(self, num_batches):
        for i, (name1, feat1) in enumerate(self.model_features.items()):
            X = self._pool_features(feat1, pool=False)
            # X_pooled = self._pool_features(feat1, pool=True)

            # frobenius norm
            # self.l2_norm[i] += torch.norm(X_pooled, p='fro', dim=-1).mean() / num_batches
            self.l2_norm[i] += torch.norm(X, p='fro', dim=-1).mean() / num_batches

            # dist = torch.cdist(X_pooled, X_pooled, p=2.0)
            dist = torch.cdist(X, X, p=2.0)

            dist_avg = (torch.sum(dist) / torch.nonzero(dist).size(0))
            self.dist_cum[i] += dist_avg / num_batches

            dist = (dist - dist.min()) / (dist.max() - dist.min())
            dist_avg_norm = (torch.sum(dist) / torch.nonzero(dist).size(0))
            self.dist_cum_norm[i] += dist_avg_norm / num_batches

            K = X @ X.t()
            K.fill_diagonal_(0.0)
            self.hsic_matrix[i, :, 0] += self._HSIC(K, K) / num_batches

            for j, (name2, feat2) in enumerate(self.model2_features.items()):
                Y = self._pool_features(feat2, pool=False)
                # Y_pooled = self._pool_features(feat2, pool=True)

                L = Y @ Y.t()
                L.fill_diagonal_(0)

                assert K.shape == L.shape, f"Feature shape mistach! {K.shape}, {L.shape}"

                self.hsic_matrix[i, j, 1] += self._HSIC(K, L) / num_batches
                self.hsic_matrix[i, j, 2] += self._HSIC(L, L) / num_batches

    def export(self) -> Dict:
        """
        Exports the CKA data along with the respective model layer names.
        :return:
        """
        return {
            "model_name": self.model_info['Name'],
            "model2_name": self.model2_info['Name'],
            "model_name_og": self.model_info['Name_og'],
            "model2_name_og": self.model2_info['Name_og'],
            "model_layers": self.model_info['Layers'],
            "model2_layers": self.model2_info['Layers'],
            "dataset_name": self.model_info['Dataset'],
            "dataset_name_og": self.model_info['Dataset_og'],
            "setting": self.model_info['Setting'],
            'l2_norm': self.l2_norm,
            "CKA": self.hsic_matrix,
            "dist": self.dist_cum,
            "dist_norm": self.dist_cum_norm,
        }

    def plot_cka(self,
                 save_path: str = None,
                 title: str = None,
                 show: bool = False):
        fig, ax = plt.subplots(figsize=(6, 5.25))
        im = ax.imshow(self.hsic_matrix, origin='lower', cmap='magma')

        ax.set_xlabel(f"Layers", fontsize=16)
        ax.set_ylabel(f"Layers", fontsize=16)

        labels = range(self.hsic_matrix.shape[0])
        ax.set_xticks(labels)
        ax.set_yticks(labels)

        if title is not None:
            ax.set_title(f"{title}", fontsize=17)
        else:
            title = f"CKA on {self.model_info['Dataset']} between \n{self.model_info['Name']} & {self.model2_info['Name']}, {self.model_info['Setting']}"
            ax.set_title(title, fontsize=17)

        add_colorbar(im)
        plt.tight_layout(pad=0.25, w_pad=0.25, h_pad=0.25)

        if save_path is not None:
            plt.savefig(save_path, dpi=300)

        if not self.debugging:
            fn = os.path.splitext(os.path.split(save_path)[-1])[0]
            wandb.log({fn: wandb.Image(fig)})

        if show:
            plt.show()

    def plot_metrics(self,
                     metric: str = 'norms',
                     save_path: str = None,
                     title: str = None,
                     show: bool = False):
        fig, ax = plt.subplots()

        if metric == 'norms':
            labels = range(self.l2_norm.shape[0])
            ax.bar(labels, self.l2_norm.cpu())
            y_label = 'L2-Norm'
        elif metric == 'dist':
            labels = range(self.dist_cum.shape[0])
            ax.bar(labels, self.dist_cum.cpu())
            y_label = 'L2-Distance'
        elif metric == 'dist_norm':
            labels = range(self.dist_cum_norm.shape[0])
            ax.bar(labels, self.dist_cum_norm.cpu())
            y_label = 'Normalized L2-Distance'

        ax.set_xlabel("Layer", fontsize=16)
        ax.set_ylabel(y_label, fontsize=16)
        ax.set_xticks(labels)

        if title is not None:
            ax.set_title(f"{title}", fontsize=17)
        else:
            title = f"{y_label} per Layer on {self.model_info['Dataset']}\n for {self.model_info['Name']}, {self.model_info['Setting']}"
            ax.set_title(title, fontsize=17)

        plt.tight_layout(pad=0.25, w_pad=0.25, h_pad=0.25)

        if save_path is not None:
            plt.savefig(save_path, dpi=300)

        if not self.debugging:
            fn = os.path.splitext(os.path.split(save_path)[-1])[0]
            wandb.log({fn: wandb.Image(fig)})

        if show:
            plt.show()


def calc_cka(results, split='train', low=[0, 4], mid=[4, 8], high=[8, 12]):
    name = 'CKA'

    ckas = {}

    results = results[name]

    levels = len(results)
    low = [0, levels // 3]
    mid = [levels // 3, 2 * (levels // 3)]
    high = [2 * (levels // 3), levels]

    for level, interval in zip(['low', 'mid', 'high'], [low, mid, high]):
        name_mean = f'cka_{level}_mean_{split}'
        name_std = f'cka_{level}_std_{split}'
        subset = results[interval[0]:interval[-1], interval[0]:interval[-1]]
        mean_val = subset.mean()
        std_val = subset.std()
        ckas.update({name_mean: mean_val, name_std: std_val})

    for i, cka in enumerate(results):
        # 1st order derivative of cka with respect to layers
        cka = cka.tolist()
        layer_change = [abs(cka[l + 1] - cka[l]) for l in range(0, len(cka) - 1)]
        try:
            layer_change_mean = mean(layer_change)
            layer_change_std = stdev(layer_change)
        except:
            layer_change_mean = np.nan
            layer_change_std = np.nan

        # 2nd order derivative of cka with respect to layers (change of layer change)
        layer_change_2nd = [abs(layer_change[i + 1] - layer_change[i]) for i in range(0, len(layer_change) - 1)]
        try:
            layer_change_2nd_mean = mean(layer_change_2nd)
            layer_change_2nd_std = stdev(layer_change_2nd)
        except:
            layer_change_2nd_mean = np.nan
            layer_change_2nd_std = np.nan

        ckas.update({
            f'{name.lower()}_change_mean_{i}_{split}': layer_change_mean,
            f'{name.lower()}_change_std_{i}_{split}': layer_change_std,
            f'{name.lower()}_change2_mean_{i}_{split}': layer_change_2nd_mean,
            f'{name.lower()}_change2_std_{i}_{split}': layer_change_2nd_std,
            })

    results = results.fill_diagonal_(0)

    for i, cka in enumerate(results):
        layer_mean = (torch.sum(cka) / torch.nonzero(cka).size(0)).item()
        ckas.update({f'{name.lower()}_{i}_{split}': layer_mean})

    overall_mean = (torch.sum(results) / torch.nonzero(results).size(0)).item()
    ckas.update({f'{name.lower()}_avg_{split}': overall_mean})

    return ckas


def calc_distances(results, split='train'):
    dists = {}
    for i, (dist, dist_norm) in enumerate(zip(results['dist'], results['dist_norm'])):
        dists.update({f'dist_{i}_{split}': dist.item(), f'dist_norm_{i}_{split}': dist_norm.item()})

    dists.update({f'dist_avg_{split}': torch.mean(results['dist']).item(),
                  f'dist_norm_avg_{split}': torch.mean(results['dist_norm']).item()})
    return dists


def calc_l2_norm(results, split='train'):
    norms = {}
    for i, norm in enumerate(results['l2_norm']):
        norms.update({f'l2_norm_{i}_{split}': norm.item()})

    norms.update({f'l2_norm_avg_{split}': torch.mean(results['l2_norm']).item()})
    return norms


def save_results_to_json(args, results_train, results_test):
    # needs to convert tensors (l2_norm, dist, dist_norm, CKA) to list
    results_train['l2_norm'] = results_train['l2_norm'].tolist()
    results_train['dist'] = results_train['dist'].tolist()
    results_train['dist_norm'] = results_train['dist_norm'].tolist()
    results_train['CKA'] = results_train['CKA'].tolist()

    results_test['l2_norm'] = results_test['l2_norm'].tolist()
    results_test['dist'] = results_test['dist'].tolist()
    results_test['dist_norm'] = results_test['dist_norm'].tolist()
    results_test['CKA'] = results_test['CKA'].tolist()

    data = {'train': results_train, 'test': results_test} 

    fp = os.path.join(args.results_dir, 'feature_metrics.json')
    with open(fp, 'w') as f:
        json.dump(data, f, indent=4)

    return 0



def build_environment_inference(args):
    set_random_seed(args.seed, numpy=True)

    train_loader, _, test_loader = build_dataloaders(args)

    model_t = build_model(args, teacher=True)
    model_s = build_model(args, student=True)

    if args.ckpt_path:
        args.setting = 'ft'
    elif args.pretrained:
        args.setting = 'pt'
    else:
        args.setting = 'scratch'

    args = adjust_args_general(args)

    if not args.debugging:
        wandb.init(config=args, project=args.project_name, entity=args.entity)
        wandb.run.name = args.run_name


    # layers teacher
    layers_t = get_layers(model_t, args.model_name_teacher, args.selector)

    # layers student
    layers_s = get_layers(model_s, args.model_name, args.selector)

    hook = FeatureMetrics(
        model_s, model_t, args.model_name, layers_s, args.model_name_teacher, layers_t,
        args.device, args.image_size, args.setting, debugging=args.debugging
    )

    amp_autocast = torch.cuda.amp.autocast if args.fp16 else suppress

    return train_loader, test_loader, hook, amp_autocast


def get_layers(model, model_name, selector=None):
    if model_name == 'vgg19_bn':
        layers = ['model.features.11', 'model.features.24', 'model.features.37', 'model.features.50']
    elif model_name == 'resnet18':
        layers = ['model.layer1.1.bn2', 'model.layer2.1.bn2', 'model.layer3.1.bn2', 'model.layer4.1.bn2']
    elif model_name == 'resnet34':
        layers = ['model.layer1.2.bn2', 'model.layer2.3.bn2', 'model.layer3.5.bn2', 'model.layer4.2.bn2']
    elif model_name == 'resnet50':
        layers = ['model.layer1.2.bn3', 'model.layer2.3.bn3', 'model.layer3.5.bn3', 'model.layer4.2.bn3']
    elif model_name == 'resnet101':
        layers = ['model.layer1.2.bn3', 'model.layer2.3.bn3', 'model.layer3.22.bn3', 'model.layer4.2.bn3']
    elif 'vit_b' in model_name:
        layers = ['model.encoder.blocks.2.norm2', 'model.encoder.blocks.5.norm2', 'model.encoder.blocks.8.norm2', 'model.encoder.blocks.11.norm2']
    elif 'beitv2_base_patch16_224_in22k' in model_name or 'deit3_base_patch16_224' in model_name:
        layers = ['model.blocks.2.norm2', 'model.blocks.5.norm2', 'model.blocks.8.norm2', 'model.blocks.11.norm2']
    elif 'deit3_large_patch16_224' in model_name:
        layers = ['model.blocks.5.norm2', 'model.blocks.11.norm2', 'model.blocks.17.norm2', 'model.blocks.23.norm2']
    elif model_name == 'van_b3':
        layers = ['model.norm1', 'model.norm2', 'model.norm3', 'model.norm4']
    elif 'convnext' in model_name:
        layers = ['model.stages.0.blocks.2.norm', 'model.stages.1.blocks.2.norm', 'model.stages.2.blocks.26.norm', 'model.stages.3.blocks.2.norm']
    elif 'convnext_base' in model_name:
        layers = ['model.stages.0.blocks.2.norm', 'model.stages.1.blocks.2.norm', 'model.stages.2.blocks.26.norm', 'model.stages.3.blocks.2.norm']
    elif 'swin' in model_name:
        layers = ['model.layers.0.blocks.1.norm2', 'model.layers.1.blocks.1.norm2', 'model.layers.2.blocks.17.norm2', 'model.layers.3.blocks.1.norm2']
    elif 'swin_base' in model_name:
        layers = ['model.layers.0.blocks.1.norm2', 'model.layers.1.blocks.1.norm2', 'model.layers.2.blocks.17.norm2', 'model.layers.3.blocks.1.norm2']
    elif 'resnetv2_101' in model_name:
        layers = ['model.stages.0.blocks.2.norm3', 'model.stages.1.blocks.3.norm3', 'model.stages.2.blocks.22.norm3', 'model.stages.3.blocks.2.norm3']
    else:
        named_modules = [name for name, _ in model.named_modules()]
        print(named_modules)
        layers = ['model']
        # for i, name in enumerate(named_modules):
        #     if i == len(named_modules) - 1:
        #         layers = [name]

    if selector == 'cal' and any([kw in model_name for kw in ('vit', 'deit', 'beit', 'swin', 'van')]):
        layers = [layer.replace('model.', 'model.encoder.0.') for layer in layers]
    elif selector == 'cal':
        layers = [layer.replace('model.', 'model.encoder.') for layer in layers]

    return layers


def compute_metrics(train_loader, test_loader, hook, amp_autocast, args):
    with torch.no_grad():
        with amp_autocast():
            hook.compare(train_loader)

            results_train = hook.export()
            hook.plot_cka(os.path.join(args.results_dir, 'cka_train.png'))
            hook.plot_metrics('norms', os.path.join(args.results_dir, 'norms_train.png'))
            hook.plot_metrics('dist', os.path.join(args.results_dir, 'dist_train.png'))
            hook.plot_metrics('dist_norm', os.path.join(args.results_dir, 'dist_norm_train.png'))

            cka_train = calc_cka(results_train, split='train')
            dists_train = calc_distances(results_train, split='train')
            norms_train = calc_l2_norm(results_train, split='train')

            hook.compare(test_loader)

            results_test = hook.export()
            hook.plot_cka(os.path.join(args.results_dir, 'cka_test.png'))
            hook.plot_metrics('norms', os.path.join(args.results_dir, 'norms_test.png'))
            hook.plot_metrics('dist', os.path.join(args.results_dir, 'dist_test.png'))
            hook.plot_metrics('dist_norm', os.path.join(args.results_dir, 'dist_norm_test.png'))

            cka_test = calc_cka(results_test, split='test')
            dists_test = calc_distances(results_test, split='test')
            norms_test = calc_l2_norm(results_test, split='test')

    log_dic = {'setting': args.setting}
    log_dic.update(cka_train)
    log_dic.update(dists_train)
    log_dic.update(norms_train)
    log_dic.update(cka_test)
    log_dic.update(dists_test)
    log_dic.update(norms_test)

    return log_dic, results_train, results_test


def main():
    args = parse_inference_args()
    args.shuffle_test = True
    args.try_fused_attn = False

    train_loader, test_loader, hook, amp_autocast = build_environment_inference(args)
    set_random_seed(args.seed, numpy=True)

    log_dic, results_train, results_test = compute_metrics(train_loader, test_loader, hook, amp_autocast, args)

    if not args.debugging:
        wandb.log(log_dic)
        wandb.finish()
    else:
        print(log_dic)

    save_results_to_json(args, results_train, results_test)

    return 0


if __name__ == '__main__':
    main()
