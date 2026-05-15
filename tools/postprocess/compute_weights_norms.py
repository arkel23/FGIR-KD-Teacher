import os
import re
import json
from warnings import warn
from typing import List, Dict
from pathlib import Path
from functools import partial
from contextlib import suppress

from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
from einops import rearrange
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
    # Resnet FSL Models
    'hiresnet50.tv_in1k': 'RN TV1', 
    'hiresnet50.tv2_in1k': 'RN TV2', 
    'hiresnet50.gluon_in1k': 'RN Gluon', 
    'hiresnet50.fb_swsl_ig1b_ft_in1k': 'RN IG1b',
    'hiresnet50.fb_ssl_yfcc100m_ft_in1k': 'RN YFCC100m',
    'hiresnet50.a1_in1k': 'RN A1',

    'hiresnet50.tv_in1k_fz': 'RN TV1', 
    'hiresnet50.tv2_in1k_fz': 'RN TV1', 
    'hiresnet50.gluon_in1k_fz': 'RN Gluon', 
    'hiresnet50.fb_swsl_ig1b_ft_in1k_fz': 'RN IG1b',
    'hiresnet50.fb_ssl_yfcc100m_ft_in1k_fz': 'RN YFCC100m',
    'hiresnet50.a1_in1k_fz': 'RN A1',

    # Resnet SSL Models
    'hiresnet50.in1k_byol': 'RN BYOL', 
    'hiresnet50.in1k_mocov3': 'RN MoCo v3', 
    'hiresnet50.in1k_spark': 'RN SparK', 
    'hiresnet50.in1k_supcon': 'RN SupCon', 
    'hiresnet50.in1k_swav': 'RN SwAV',
    'hiresnet50.in21k_miil': 'RN IN21k-P',

    'hiresnet50.in1k_byol_fz': 'RN BYOL', 
    'hiresnet50.in1k_mocov3_fz': 'RN MoCo v3', 
    'hiresnet50.in1k_spark_fz': 'RN SparK', 
    'hiresnet50.in1k_supcon_fz': 'RN SupCon', 
    'hiresnet50.in1k_swav_fz': 'RN SwAV',
    'hiresnet50.in21k_miil_fz': 'RN IN21k-P',

    # ViT FSL Models
    'hivit_base_patch16_224.orig_in21k': 'ViT',
    'hideit_base_patch16_224.fb_in1k': 'DeiT',
    'hivit_base_patch16_224_miil.in21k': 'ViT IN21k-P',
    'hideit3_base_patch16_224.fb_in22k_ft_in1k': 'DeiT 3 (IN21k)',
    'hideit3_base_patch16_224.fb_in1k': 'DeiT 3 (IN1k)',

    'hivit_base_patch16_224.orig_in21k_fz': 'ViT',
    'hideit_base_patch16_224.fb_in1k_fz': 'DeiT',
    'hivit_base_patch16_224_miil.in21k_fz': 'ViT IN21k-P',
    'hideit3_base_patch16_224.fb_in22k_ft_in1k_fz': 'DeiT 3 (IN21k)',
    'hideit3_base_patch16_224.fb_in1k_fz': 'DeiT 3 (IN1k)',

    # ViT SSL Models
    'hivit_base_patch16_224.in1k_mocov3': 'ViT MoCo v3',
    'hivit_base_patch16_224.dino': 'ViT DINO',
    'hivit_base_patch16_224.mae': 'ViT MAE',
    'hivit_base_patch16_clip_224.laion2b': 'ViT CLIP',
    'hivit_base_patch16_siglip_224.v2_webli': 'ViT SigLIP v2',

    'hivit_base_patch16_224.in1k_mocov3_fz': 'ViT MoCo v3',
    'hivit_base_patch16_224.dino_fz': 'ViT DINO',
    'hivit_base_patch16_224.mae_fz': 'ViT MAE',
    'hivit_base_patch16_clip_224.laion2b_fz': 'ViT CLIP',
    'hivit_base_patch16_siglip_224.v2_webli_fz': 'ViT SigLIP v2',
}


SETTINGS_DIC = {
    'scratch': '(Random Init)',
    'fz': '(Frozen)',
    'ft': '(Fine-Tuned)',
}


def adjust_args_general(args):
    args.run_name = '{}_{}'.format(
        args.model_name, args.serial
    )

    args.results_dir = os.path.join(args.results_inference, args.run_name)
    os.makedirs(args.results_dir, exist_ok=True)

    return args


class FeatureMetrics:
    def __init__(
        self,
        model: nn.Module,
        model_name: str = None,
        model_layers: List[str] = None,
        device: str ='cpu',
        setting: str = 'fz',
        debugging: bool = False
    ):
        """

        :param model: (nn.Module) Neural Network 1
        :param model_name: (str) Name of model 1
        :param model_layers: (List) List of layers to extract features from
        :param device: Device to run the model
        """

        self.model = model

        self.device = device

        self.model_info = {}

        self.model_info['Setting'] = SETTINGS_DIC.get(setting, setting)

        if model_name is None:
            self.model_info['Name_og'] = model.__repr__().split('(')[0]
        else:
            self.model_info['Name_og'] = model_name
        self.model_info['Name'] = MODELS_DIC.get(self.model_info['Name_og'], self.model_info['Name_og'])

        self.model_info['Layers'] = model_layers

        self.model_features = {}

        if len(list(model.modules())) > 150 and model_layers is None:
            warn("Model 1 seems to have a lot of layers. " \
                 "Consider giving a list of layers whose features you are concerned with " \
                 "through the 'model_layers' parameter. Your CPU/GPU will thank you :)")

        self.model_layers = model_layers

        self.model = self.model.to(self.device)

        self.model.eval()

        self.debugging = debugging

        print(self.model_info)

    def compare(self) -> None:
        """
        Computes the weight magnitudes for given model.
        """

        layers = self.model_layers if self.model_layers is not None else list(self.model.modules())

        N = len(layers)

        self.weights_l2_norm = torch.zeros(N, device=self.device)
        self.weights = {}
        self.bn_means = {}
        self.bn_vars = {}

        self.compute_norms()

    def compute_norms(self):
        for i, layer in enumerate(self.model_layers):
            weights = self.model.state_dict()[layer]

            # rearrange into 1d
            weights = torch.flatten(weights)

            # compute norms
            self.weights_l2_norm[i] = torch.norm(weights, p='fro', dim=-1).detach().cpu()

            # store weights
            self.weights[layer] = weights.detach().cpu()

            if 'layer' in layer and 'bn' in layer:
                stage = re.search(r"layer\d+", layer).group()
                index = re.search(r"(?<=\.)\d+(?=\.)", layer).group()
                bn = re.search(r"bn\d+", layer).group()

                block = getattr(self.model.model, f'{stage}')
                block = block[int(index)]
                block = getattr(block, bn)

                self.bn_means[layer] = block.running_mean.detach().cpu()
                self.bn_vars[layer] = block.running_var.detach().cpu()

            print(layer, self.model.state_dict()[layer].shape, weights.shape, self.weights_l2_norm[i])

    def export(self) -> Dict:
        """
        Exports the data along with the respective model layer names.
        :return:
        """
        return {
            "model_name": self.model_info['Name'],
            "model_name_og": self.model_info['Name_og'],
            "model_layers": self.model_info['Layers'],
            "setting": self.model_info['Setting'],
            'weights_l2_norm': self.weights_l2_norm,
        }

    def plot_metrics(self,
                     metric: str = 'norms',
                     save_path: str = None,
                     title: str = None,
                     show: bool = False):
        fig, ax = plt.subplots()

        if metric == 'norms':
            labels = range(self.weights_l2_norm.shape[0])
            ax.bar(labels, self.weights_l2_norm.cpu())
            y_label = 'Weights L2-Norm'
        elif 'weights_' in metric:
            metric_ = metric.replace('weights_', '')
            ax.hist(self.weights[metric_])
            y_label = f'Weights Histogram'
        elif 'bn_means_' in metric:
            metric_ = metric.replace('bn_means_', '')
            ax.hist(self.bn_means[metric_])
            y_label = f'BatchNorm Channel Mean Histogram'
        elif 'bn_vars_' in metric:
            metric_ = metric.replace('bn_vars_', '')
            ax.hist(self.bn_vars[metric_])
            y_label = f'BatchNorm Channel Variance Histogram'

        if any([kw in metric for kw in ['weights_', 'bn_means', 'bn_vars']]):
            ax.set_xlabel("Magnitude", fontsize=16)
            ax.set_ylabel('Counts', fontsize=16)
        else:
            ax.set_xlabel("Layer", fontsize=16)
            ax.set_ylabel(y_label, fontsize=16)
            # ax.set_xticks(labels)

        if title is not None:
            ax.set_title(f"{title}", fontsize=17)
        else:
            title = f"{y_label} for\n{self.model_info['Name']} {self.model_info['Setting']}"
            ax.set_title(title, fontsize=17)

        plt.tight_layout(pad=0.25, w_pad=0.25, h_pad=0.25)

        if save_path is not None:
            plt.savefig(save_path, dpi=300)

        if not self.debugging:
            fn = os.path.splitext(os.path.split(save_path)[-1])[0]
            wandb.log({fn: wandb.Image(fig)})

        if show:
            plt.show()

        plt.close()


def calc_weights_l2_norm(results):
    norms = {}
    for i, norm in enumerate(results['weights_l2_norm']):
        norms.update({f'weights_l2_norm_{i}': norm.item()})

    norms.update({f'weights_l2_norm_avg': torch.mean(results['weights_l2_norm']).item()})
    return norms


def save_results_to_json(args, results):
    # needs to convert tensors (weights_l2_norm) to list
    results['weights_l2_norm'] = results['weights_l2_norm'].tolist()

    data = {'weights': results} 

    fp = os.path.join(args.results_dir, 'weights_metrics.json')
    with open(fp, 'w') as f:
        json.dump(data, f, indent=4)

    return 0


def setup_environment(args):
    set_random_seed(args.seed, numpy=True)

    # dataloaders
    args.shuffle_test = True
    train_loader, val_loader, test_loader = build_dataloaders(args)

    model = build_model(args)

    if args.ckpt_path:
        args.setting = 'ft'
    elif args.pretrained:
        args.setting = 'fz'
    else:
        args.setting = 'scratch'

    args = adjust_args_general(args)

    if not args.debugging:
        wandb.init(config=args, project=args.project_name, entity=args.entity)
        wandb.run.name = args.run_name

    layers = []
    for name, _ in model.named_parameters():
        # print(name)
        if args.weights_type == 'norm':
            if 'resnet' in args.model_name and 'bn' in name and 'weight' in name:
                layers.append(name)
            elif ('deit' in args.model_name or 'vit' in args.model_name) and (any(
                [kw in name for kw in ['norm']]) and 'weight' in name):
                layers.append(name)

        elif args.weights_type == 'main':
            if 'resnet' in args.model_name and 'conv' in name and 'weight' in name:
                layers.append(name)
            elif ('deit' in args.model_name or 'vit' in args.model_name) and (any(
                [kw in name for kw in ['attn', 'mlp']]) and 'weight' in name):
                layers.append(name)

        else:
            if 'weight' in name:
                layers.append(name)

    extractor = FeatureMetrics(model, args.model_name, layers, args.device,
                          args.setting, debugging=args.debugging,)

    return extractor


def main():
    args = parse_inference_args()

    extractor = setup_environment(args)

    amp_autocast = torch.cuda.amp.autocast if args.fp16 else suppress

    with torch.no_grad():
        with amp_autocast():
            extractor.compare()

            results = extractor.export()
            extractor.plot_metrics('norms', os.path.join(args.results_dir, 'norms.png'))

            for layer in extractor.model_layers:
                extractor.plot_metrics(f'weights_{layer}', os.path.join(args.results_dir, f'weights_{layer}.png'))

                if 'layer' in layer and 'bn' in layer:
                    extractor.plot_metrics(f'bn_means_{layer}', os.path.join(args.results_dir, f'bn_means_{layer}.png'))
                    extractor.plot_metrics(f'bn_vars_{layer}', os.path.join(args.results_dir, f'bn_vars_{layer}.png'))

            norms = calc_weights_l2_norm(results)


    log_dic = {'setting': args.setting}
    log_dic.update(norms)

    if not args.debugging:
        wandb.log(log_dic)
        wandb.finish()
    else:
        print(log_dic)

    save_results_to_json(args, results)

    return 0


if __name__ == '__main__':
    main()
