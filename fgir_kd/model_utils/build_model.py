import re
from copy import deepcopy
from types import SimpleNamespace

import timm
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .modules_others import van_dict, ViT, ViTConfig, Head, CAL, CRD


VITS = [
    'vit_n16', 'vit_m16', 'vit_t4', 'vit_t8', 'vit_t16', 'vit_t32',
    'vit_s8', 'vit_s16', 'vit_s32',
    'vit_b8', 'vit_b16', 'vit_b32', 'vit_l16', 'vit_l32', 'vit_h14']


def build_model(args, teacher=False, student=False):
    if teacher:
        if args.legacy_swin_teacher:
            from fgir_kd.model_utils.modules_others.swin import swin_base_patch4_window7_224, swin_large_patch4_window7_224
        args = deepcopy(args)
        args.model_name = args.model_name_teacher
        args.ckpt_path = args.ckpt_path_teacher
    elif student:
        args = deepcopy(args)
        args.image_size = args.student_image_size if args.student_image_size else args.image_size

    # initiates model and loss
    if (args.model_name in VITS or 'van' in args.model_name or
        args.model_name in timm.list_models() or args.model_name in timm.list_models(pretrained=True)):
        model = ClassifierModel(args, teacher, student)
    else:
        raise NotImplementedError

    if args.ckpt_path:
        load_model_compatibility_mode(args, model)

    if teacher and not args.train_both:
        freeze_backbone(model)

    if args.distributed:
        model.cuda()
    else:
        model.to(args.device)

    if args.distributed and (not teacher or args.train_both):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    model.zero_grad()

    if teacher and args.teacher_eval_mode:
        model.eval()
    elif teacher:
        model.train()
    else:
        model.eval()

    print(f'Initialized classifier: {args.model_name}')
    return model


def freeze_backbone(model):
    for name, param in model.named_parameters():
        param.requires_grad = False

    print('Total parameters (M): ', sum([p.numel() for p in model.parameters()]) / (1e6))
    print('Trainable parameters (M): ', sum([p.numel() for p in model.parameters() if p.requires_grad]) / (1e6))
    return 0


def convert_cal_student(state_dict, drop_head=True):
    new_state_dict = {}
    for k, v in state_dict.items():

        if 'dfsm.1.' in k and not drop_head:
            k = k.replace('model.dfsm.1.', 'head.head.')

        elif 'dfsm.' in k:
            # expected_missing_keys += ['model.dfsm.1.weight', 'model.dfsm.1.bias']
            continue

        if 'model.encoder.0' in k:
            new_k = k.replace('model.encoder.0.', 'model.')
        else:
            new_k = k.replace('model.encoder.', 'model.')
        new_state_dict[new_k] = v

    return new_state_dict


def load_model_compatibility_mode(args, model):
    state_dict = torch.load(
        args.ckpt_path, map_location=torch.device('cpu'))['model']
    expected_missing_keys = []

    # retrocompatibility with prev experiments
    if 'model.head.head.weight' in state_dict.keys():
        state_dict['head.head.weight'] = state_dict.pop('model.head.head.weight')
        state_dict['head.head.bias'] = state_dict.pop('model.head.head.bias')
    elif 'head.weight' in state_dict.keys():
        state_dict['head.head.weight'] = state_dict.pop('head.weight')
        state_dict['head.head.bias'] = state_dict.pop('head.bias')

    # saved when using distributed training has an additional module. at the start of the keys
    if list(state_dict.keys())[0].startswith('module.'):
        for k in list(state_dict.keys()):
            if k.startswith('module.'):
                new_k = k.replace('module.', '', 1)
                state_dict[new_k] = state_dict.pop(k)

    if args.convert_cal_student_keep_head:
        state_dict = convert_cal_student(state_dict, drop_head=False)
    elif args.transfer_learning_cal:
        state_dict = convert_cal_student(state_dict)

    if args.transfer_learning:
        # modifications to load partial state dict
        if ('model.head.weight' in state_dict):
            expected_missing_keys += ['model.head.weight', 'model.head.bias']
        for key in expected_missing_keys:
            state_dict.pop(key)
    ret = model.load_state_dict(state_dict, strict=False)
    print('''Missing keys when loading pretrained weights: {}
            Expected missing keys: {}'''.format(ret.missing_keys, expected_missing_keys))
    print('Unexpected keys when loading pretrained weights: {}'.format(
        ret.unexpected_keys))
    print('Loaded from custom checkpoint.')
    return 0



def get_backbone(args):
    args.classifier = 'pool'  # def, use cls for vit/deit if not cal

    if 'vitfs' in args.model_name:
        model = timm.create_model(
            args.model_name, pretrained=False, num_classes=0, img_size=args.image_size,
            drop_path_rate=args.sd, global_pool='', args=args)

    elif args.model_name in VITS:
        if not (args.selector == 'cal' or args.transfer_learning_cal or args.convert_cal_student_keep_head):
            args.classifier = 'cls'
        # init default config
        cfg = ViTConfig(model_name=args.model_name, image_size=args.image_size)
        cfg.classifier = args.classifier
        cfg.calc_dims()

        # init model
        model = ViT(cfg, pretrained=args.pretrained)

    elif 'levit' in args.model_name or 'swin' in args.model_name:
        model = timm.create_model(
            args.model_name, pretrained=args.pretrained, num_classes=0,
            img_size=args.image_size, drop_path_rate=args.sd, global_pool='')
    elif 'deit' in args.model_name or 'vit' in args.model_name:
        if not (args.selector == 'cal' or args.transfer_learning_cal):
            args.classifier = 'cls'
        cls = True if args.classifier == 'cls' else False
        model = timm.create_model(
            args.model_name, pretrained=args.pretrained, num_classes=0, class_token=cls,
            img_size=args.image_size, drop_path_rate=args.sd, global_pool='')
    elif 'van' in args.model_name:
        model = van_dict[args.model_name](
            pretrained=args.pretrained,img_size=args.image_size, drop_path_rate=args.sd)
    elif 'vgg' in args.model_name:
        model = timm.create_model(args.model_name, pretrained=args.pretrained,
                                  num_classes=0, global_pool='',
                                  pre_logits=False if args.selector == 'cal' else True)
    elif 'lrnet' in args.model_name:
        model = timm.create_model(
            args.model_name, pretrained=False, num_classes=0,
            drop_path_rate=args.sd, global_pool='')
    else:
        model = timm.create_model(
            args.model_name, pretrained=args.pretrained, num_classes=0,
            drop_path_rate=args.sd, global_pool='')

    return model


def get_layers(args, model, teacher=False, student=False):
    if teacher:
        model_name = args.model_name_teacher
        layer_names = args.layer_names
        num_layers = args.num_layers
    elif student:
        model_name = args.model_name
        layer_names = args.layer_names_student
        num_layers = args.num_layers_student

    if layer_names:
        layer_names = layer_names[-num_layers:]
        if teacher:
            args.layer_names = layer_names
        elif student:
            args.layer_names_student = layer_names
        return layer_names

    if 'convnext' in model_name or 'resnetv2' in model_name:
        pattern = re.compile(r'.*stages\.\d+\.blocks\.\d+$')

    elif 'resnet' in model_name:
        pattern = re.compile(r'.*layer\d+\.\d+$')

    elif 'swin' in model_name:
        pattern = re.compile(r'.*layers\.\d+\.blocks\.\d+$')

    elif 'beitv2' in model_name or 'deit' in model_name:
        pattern = re.compile(r'.*blocks\.\d+$')

    elif model_name in VITS:
        pattern = re.compile(r'.*encoder.blocks\.\d+$')

    elif 'vit' in model_name:
        pattern = re.compile(r'.*blocks\.\d+$')

    elif 'van' in model_name:
        pattern = re.compile(r'.*block\d+\.\d+')

    elif 'vgg' in model_name:
        pattern = re.compile(r'.*features\.\d+')

    elif 'lcnet' in model_name:
        pattern = re.compile(r'.*blocks\d+\.\d+$')

    else:
        return None
        # raise NotImplementedError
    
    all_names = []
    for name, _ in model.named_modules():
        all_names.append(name)

    layers = [l for l in all_names if pattern.match(l)]

    if args.selector == 'cal' and any([kw in args.model_name for kw in ('vit', 'deit', 'beit', 'swin', 'van')]):
        layers = ['encoder.0.' + l for l in layers]
        layers.append('encoder.1')
    elif args.selector == 'cal':
        layers = ['encoder.' + l for l in layers]
        layers.append('encoder')

    print('Stage output layers in model: ', layers)

    layers = layers[-num_layers:]
    if teacher:
        args.layer_names = layers
    elif student:
        args.layer_names_student = layers

    print('Layers to use: ', layers)

    return layers


class ClassifierModel(nn.Module):
    def __init__(self, args, teacher=False, student=False):
        super(ClassifierModel, self).__init__()

        model = get_backbone(args)
        img_size = args.student_image_size if (args.student_image_size and student) else args.image_size
        s, d, shape_format = self.get_out_features(img_size, model)

        if teacher:
            layers = get_layers(args, model, teacher=True)
        elif student:
            layers = get_layers(args, model, student=True)
        else:
            layers = None

        if args.selector == 'cal':
            self.model = CAL(
                model=model,
                seq_len=s,
                output_size=d,
                num_classes=args.num_classes,
                shape_format=shape_format,
                device=args.device,
                teacher=teacher,
                student=student,
                tgda=args.tgda,
                kd_aux_loss=args.kd_aux_loss,
                num_images=args.num_images_train,
                cont_negatives=args.cont_negatives,
                cont_temp=args.cont_temp,
                cont_loss=args.cont_loss,
                image_size=args.image_size,
                layers=layers,
                pooling_function=args.pooling_function,
                pool_size=args.pool_size,
                disc_feats_norm=args.disc_feats_norm,
                disc_feats_sign_sqrt=args.disc_feats_sign_sqrt,
                if_channels=args.if_channels,
                mlp_ratio=args.mlp_ratio,
                pre_resize_factor=args.pre_resize_factor,
                student_image_size=args.student_image_size,
                cal_ap_only=args.cal_ap_only,
                num_aug_maps=args.num_aug_maps,
                cam_aug=args.cam_aug,
                train_both=args.train_both,
                teacher_loss=args.teacher_loss,
            )

        else:
            self.model = model
            self.head = Head(args.classifier, d, args.num_classes, shape_format)

            if teacher and args.kd_aux_loss == 'crd':
                self.return_inter_feats = True
                self.if_channels = d
            elif student and args.kd_aux_loss == 'crd':
                self.loss_weight = args.loss_kd_aux_weight
                self.crd = CRD(
                    args.num_images_train, d, args.if_channels, int(d * args.mlp_ratio),
                    'gap', shape_format, args.cont_negatives, args.cont_temp
                )

        self.cfg = SimpleNamespace(**{'seq_len': s, 'hidden_size': d})

    @torch.no_grad()
    def get_out_features(self, image_size, model):
        x = torch.rand(2, 3, image_size, image_size)
        x = model(x)

        if len(x.shape) == 3:
            b, s, d = x.shape
            shape_format = 'bsd'
        elif len(x.shape) == 4:
            b, d, h, w = x.shape

            if h == w:
                shape_format = 'bdhw'
            elif h != w:
                # b, h, w, c is the actual shape
                shape_format = 'bhwd'
                temp1 = d
                temp2 = h
                temp3 = w
                h = temp1
                w = temp2
                d = temp3

            s = h * w

        print('Output feature shape: ', x.shape)

        return s, d, shape_format

    def forward(self, images, targets=None, output_t=None, idx=None, sample_idx=None):
        if hasattr(self, 'head'):
            features = self.model(images)
            out = self.head(features)

            if hasattr(self, 'return_inter_feats') and self.training:
                return out, features
            elif hasattr(self, 'crd') and self.training:
                loss = self.crd(features, output_t[1], idx, sample_idx)
                return out, loss

            return out

        else:
            out = self.model(images, targets, output_t, idx, sample_idx)

        return out
