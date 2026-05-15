import os
import json
from warnings import warn
from typing import List, Dict
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
    'soyageingr1': 'SoyAgeR1',
    'soyageingr3': 'SoyAgeR3',
    'soyageingr4': 'SoyAgeR4',
    'soyageingr5': 'SoyAgeR5',
    'soyageingr6': 'SoyAgeR6',
    'soygene': 'SoyGene',
    'soyglobal': 'SoyGlobal',
    'soylocal': 'SoyLocal',
    'vegfru': 'VegFru',
}

SETTINGS_DIC = {
    'scratch': '(Random Init)',
    'fz': '(Frozen)',
    'ft': '(Fine-Tuned)',
}


def adjust_args_general(args):
    args.run_name = '{}_{}_{}'.format(
        args.dataset_name, args.model_name, args.serial
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
        out_size: int = 7,
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

        self.model_info['Layers'] = []

        self.model_features = {}

        if len(list(model.modules())) > 150 and model_layers is None:
            warn("Model 1 seems to have a lot of layers. " \
                 "Consider giving a list of layers whose features you are concerned with " \
                 "through the 'model_layers' parameter. Your CPU/GPU will thank you :)")

        self.model_layers = model_layers

        self._insert_hooks()
        self.model = self.model.to(self.device)

        self.model.eval()

        self.pool = nn.AdaptiveAvgPool2d((out_size, out_size)).to(self.device)

        self.debugging = debugging

        print(self.model_info)

    def _log_layer(self,
                   model: str,
                   name: str,
                   layer: nn.Module,
                   inp: torch.Tensor,
                   out: torch.Tensor):

        if model == "model":
            self.model_features[name] = out
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

    def _pool_features(self, feat, batch_flatten=False, pool=False):
        if batch_flatten:
            pooled = feat.flatten(1)

        if pool:
            if len(feat.shape) == 2:
                pooled = feat

            elif len(feat.shape) == 3:
                h = int(feat.shape[1] ** 0.5)
                if h ** 2 == feat.shape[1]:
                    pooled = rearrange(feat, 'b (h w) d -> b d h w', h=h)
                    pooled = self.pool(pooled)
                    pooled = rearrange(pooled, 'b c h w -> (b h w) c')  
                else:
                    x_cls, x_others = torch.split(feat, [1, int(h**2)], dim=1)
                    x_others = rearrange(x_others, 'b (h w) d -> b d h w', h=h)
                    x_others = self.pool(x_others)
                    x_others = rearrange(x_others, 'b d h w -> b (h w) d')
                    pooled = torch.cat([x_cls, x_others], dim=1)
                    pooled = rearrange(pooled, 'b s d -> (b s) d')

            elif len(feat.shape) == 4:
                b, c, h, w = feat.shape
                if h != w:
                    feat = rearrange(feat, 'b h w c -> b c h w')
                pooled = self.pool(feat)
                pooled = rearrange(pooled, 'b c h w -> (b h w) c')

        else:
            if len(feat.shape) == 2:
                pooled = feat
            elif len(feat.shape) == 3:
                pooled = rearrange(feat, 'b s d -> (b s) d')
            elif len(feat.shape) == 4:
                b, c, h, w = feat.shape
                if h == w:
                    pooled = rearrange(feat, 'b c h w -> (b h w) c')
                else:
                    pooled = rearrange(feat, 'b h w c -> (b h w) c')

 
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

        N = len(layers)

        self.dist_cum = torch.zeros(N, device=self.device)
        self.dist_cum_norm = torch.zeros(N, device=self.device)
        self.l2_norm = torch.zeros(N, device=self.device)

        self.norms = {}

        # images, *_ = next(iter(dataloader1))
        # images = images.to(self.device)
        # _ = self.model(images)

        # feats = [self._pool_features(v, pool=True) for v in self.model_features.values()]
        # ft_shapes = [ft.shape for ft in feats]
        # norm_shapes = [torch.norm(ft, dim=-1).shape for ft in feats]

        # self.features = {k: torch.empty(v, device='cpu') for k, v in zip(self.model_features.keys, ft_shapes)}
        # self.norms = {k: torch.empty(v, device='cpu') for k, v in zip(self.model_features.keys, norm_shapes)}

        # print([ft.shape for ft in self.features.values()])
        # print([ft.shape for ft in self.norms.values()])

        num_batches = len(dataloader1)

        for (x1, *_) in tqdm(dataloader1, desc="| Comparing features |", total=num_batches):

            self.model_features = {}
            x1 = x1.to(self.device)
            _ = self.model(x1)

            self.compare_l2_dist(num_batches)

        # print([ft.shape for ft in self.norms.values()])

    def compare_l2_dist(self, num_batches):
        for i, (name1, feat1) in enumerate(self.model_features.items()):
            # BxCxHxW or BxSxD -> (BHW) x C
            x = self._pool_features(feat1, pool=False)
            x_pooled = self._pool_features(feat1, pool=True)

            # compute norms
            norms = torch.norm(x, p='fro', dim=-1).detach().cpu()

            # store features
            if name1 in self.norms.keys():
                self.norms[name1] = torch.cat([self.norms[name1], norms], dim=0)
            else:
                self.norms[name1] = norms

            # frobenius norm
            self.l2_norm[i] += torch.norm(x, p='fro', dim=-1).mean() / num_batches

            dist = torch.cdist(x_pooled, x_pooled, p=2.0)

            dist_avg = (torch.sum(dist) / torch.nonzero(dist).size(0))
            self.dist_cum[i] += dist_avg / num_batches

            dist = (dist - dist.min()) / (dist.max() - dist.min())
            dist_avg_norm = (torch.sum(dist) / torch.nonzero(dist).size(0))
            self.dist_cum_norm[i] += dist_avg_norm / num_batches

    def export(self) -> Dict:
        """
        Exports the data along with the respective model layer names.
        :return:
        """
        return {
            "model_name": self.model_info['Name'],
            "model_name_og": self.model_info['Name_og'],
            "model_layers": self.model_info['Layers'],
            "dataset_name": self.model_info['Dataset'],
            "dataset_name_og": self.model_info['Dataset_og'],
            "setting": self.model_info['Setting'],
            'l2_norm': self.l2_norm,
            "dist": self.dist_cum,
            "dist_norm": self.dist_cum_norm,
        }

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
            y_label = 'Normalized L2-Dist.'
        elif 'norms_' in metric:
            metric_ = metric.replace('norms_', '')
            ax.hist(self.norms[metric_])
            y_label = f'L2-Norm Histogram'

        if 'norms_' in metric:
            ax.set_xlabel("L2-Norm Magnitudes", fontsize=16)
            ax.set_ylabel('Counts', fontsize=16)
        else:
            ax.set_xlabel("Layer", fontsize=16)
            ax.set_ylabel(y_label, fontsize=16)
            ax.set_xticks(labels)

        if title is not None:
            ax.set_title(f"{title}", fontsize=17)
        else:
            title = f"{y_label} per Layer on {self.model_info['Dataset']}\n for {self.model_info['Name']} {self.model_info['Setting']}"
            ax.set_title(title, fontsize=17)

        plt.tight_layout(pad=0.25, w_pad=0.25, h_pad=0.25)

        if save_path is not None:
            plt.savefig(save_path, dpi=300)

        if not self.debugging:
            fn = os.path.splitext(os.path.split(save_path)[-1])[0]
            wandb.log({fn: wandb.Image(fig)})

        if show:
            plt.show()


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
    # needs to convert tensors (l2_norm, dist, dist_norm) to list
    results_train['l2_norm'] = results_train['l2_norm'].tolist()
    results_train['dist'] = results_train['dist'].tolist()
    results_train['dist_norm'] = results_train['dist_norm'].tolist()

    results_test['l2_norm'] = results_test['l2_norm'].tolist()
    results_test['dist'] = results_test['dist'].tolist()
    results_test['dist_norm'] = results_test['dist_norm'].tolist()

    data = {'train': results_train, 'test': results_test} 

    fp = os.path.join(args.results_dir, 'feature_metrics.json')
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
    for name, _ in model.named_modules():
        # print(name)
        if 'resnet' in args.model_name and 'bn2' in name:
            layers.append(name)
        elif ('deit' in args.model_name or 'vit' in args.model_name) and (any(
            [kw in name for kw in ['norm2']])):
            layers.append(name)

    feature_metrics = FeatureMetrics(model, args.model_name, layers, args.device,
                          args.setting, debugging=args.debugging,)

    return train_loader, test_loader, feature_metrics


def main():
    args = parse_inference_args()

    train_loader, test_loader, feature_metrics = setup_environment(args)

    amp_autocast = torch.cuda.amp.autocast if args.fp16 else suppress

    with torch.no_grad():
        with amp_autocast():
            feature_metrics.compare(train_loader)

            results_train = feature_metrics.export()
            feature_metrics.plot_metrics('norms', os.path.join(args.results_dir, 'norms_train.png'))
            feature_metrics.plot_metrics('dist', os.path.join(args.results_dir, 'dist_train.png'))
            feature_metrics.plot_metrics('dist_norm', os.path.join(args.results_dir, 'dist_norm_train.png'))

            for layer in feature_metrics.model_layers:
                feature_metrics.plot_metrics(f'norms_{layer}', os.path.join(args.results_dir, f'norms_{layer}_train.png'))

            dists_train = calc_distances(results_train, split='train')
            norms_train = calc_l2_norm(results_train, split='train')

            feature_metrics.compare(test_loader)

            results_test = feature_metrics.export()
            feature_metrics.plot_metrics('norms', os.path.join(args.results_dir, 'norms_test.png'))
            feature_metrics.plot_metrics('dist', os.path.join(args.results_dir, 'dist_test.png'))
            feature_metrics.plot_metrics('dist_norm', os.path.join(args.results_dir, 'dist_norm_test.png'))

            for layer in feature_metrics.model_layers:
                feature_metrics.plot_metrics(f'norms_{layer}', os.path.join(args.results_dir, f'norms_{layer}_test.png'))

            dists_test = calc_distances(results_test, split='test')
            norms_test = calc_l2_norm(results_test, split='test')

    log_dic = {'setting': args.setting}
    log_dic.update(dists_train)
    log_dic.update(norms_train)
    log_dic.update(dists_test)
    log_dic.update(norms_test)

    if not args.debugging:
        wandb.log(log_dic)
        wandb.finish()
    else:
        print(log_dic)

    save_results_to_json(args, results_train, results_test)

    return 0


if __name__ == '__main__':
    main()
