import os
import argparse
import torch

from .yaml_config_hook import yaml_config_hook


VITS = ['vit_t4', 'vit_t8', 'vit_t16', 'vit_t32', 'vit_s8', 'vit_s16', 'vit_s32',
        'vit_b8', 'vit_b16', 'vit_b32', 'vit_l16', 'vit_l32', 'vit_h14']
MODELS = VITS
VIS_LIST = ('gradcam')


def add_adjust_common_dependent(args):
    if args.vis_errors_save:
        args.vis_errors = True

    if args.vis_errors:
        args.test_only_bs1 = True

    if args.test_only_bs1:
        print('When using test only changes batch size to 1 to simulate streaming')
        args.test_only = True
        args.batch_size = 1

    args.effective_batch_size = args.batch_size * args.gradient_accumulation_steps

    if args.base_lr:
        args.lr = args.base_lr * (args.effective_batch_size / 8)

    # distributed
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    if args.distributed:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        args.device = torch.device(f'cuda:{args.local_rank}')
        torch.cuda.set_device(args.device)

        args.effective_batch_size = args.effective_batch_size * args.world_size

        if args.base_lr:
            args.lr = args.base_lr * (args.effective_batch_size / 8)

    if not args.resize_size:
        args.resize_size = int(args.image_size * 1.34)

    if not args.test_resize_size:
        args.test_resize_size = args.resize_size

    assert not (args.distributed and args.test_only), 'test_only cannot be used with multi gpu'

    args.loss_kd_weight_actual = args.loss_kd_weight

    return args


def add_common_args():
    parser = argparse.ArgumentParser('Arguments for code: FGIRKD')
    # general
    parser.add_argument('--project_name', type=str, default='KD_TGDA',
                        help='project name for wandb')
    parser.add_argument('--entity', type=str, default='nycu_pcs', help='wandb entity')
    parser.add_argument('--debugging', action='store_true',
                        help='when true disables wandb and exits after a single pass')
    parser.add_argument('--serial', default=0, type=int, help='serial number for run')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs')
    parser.add_argument('--log_freq', type=int,
                        default=100, help='print frequency (iters)')
    parser.add_argument('--save_freq', type=int,
                        default=200, help='save frequency (epochs)')
    parser.add_argument('--results_dir', type=str, default='results_train',
                        help='dir to save models from base_path')
    parser.add_argument('--per_class_acc_results', type=str, default='per_class_acc.csv')
    parser.add_argument('--ind_preds_results', type=str, default='ind_preds.csv')
    parser.add_argument('--plot_gradients', action='store_true', help='may need to turn of fp16')
    return parser


def add_data_args(parser):
    parser.add_argument('--image_size', type=int, default=224, help='image_size')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--cpu_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--prefetch_factor', type=int, default=None)
    parser.add_argument('--custom_mean_std', action='store_true', help='custom mean/std')
    parser.add_argument('--dataset_in_memory', action='store_true',
                        help='load whole dataset into RAM to reduce reads from storage')
    parser.add_argument('--pin_memory', action='store_false', help='pin memory for gpu (def: true)')
    # dataset
    parser.add_argument('--shuffle_test', action='store_true',
                        help='if true then shuffles test data')
    parser.add_argument('--dataset_name', default=None, type=str, help='dataset name')
    parser.add_argument('--dataset_root_path', type=str, default=None,
                        help='the root directory for where the data/feature/label files are')
    # folders with images (can be same: those where it's all stored in 'data')
    parser.add_argument('--folder_train', type=str, default='data',
                        help='the directory where images are stored, ex: dataset_root_path/train/')
    parser.add_argument('--folder_val', type=str, default='data',
                        help='the directory where images are stored, ex: dataset_root_path/val/')
    parser.add_argument('--folder_test', type=str, default='data',
                        help='the directory where images are stored, ex: dataset_root_path/test/')
    # df files with img_dir, class_id
    parser.add_argument('--df_train', type=str, default='train.csv',
                        help='the df csv with img_dirs, targets, def: train.csv')
    parser.add_argument('--df_trainval', type=str, default='train_val.csv',
                        help='the df csv with img_dirs, targets, def: train_val.csv')
    parser.add_argument('--df_val', type=str, default='val.csv',
                        help='the df csv with img_dirs, targets, def: val.csv')
    parser.add_argument('--df_test', type=str, default='test.csv',
                        help='the df csv with img_dirs, targets, root/test.csv')
    parser.add_argument('--df_classid_classname', type=str, default='classid_classname.csv',
                        help='the df csv with classnames and class ids, root/classid_classname.csv')

    parser.add_argument('--video', action='store_true',
                        help='if true means we will use video dataset/transform/model')
    return parser


def add_optim_scheduler_args(parser):
    # optimizer
    parser.add_argument('--opt', default='sgd', type=str,
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--base_lr', type=float, default=None,
                        help='base_lr if using scaling lr (lr = base_lr * bs/256')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--opt_eps', default=1e-8, type=float,
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')
    # fp 16 stability and gradient accumulation
    parser.add_argument('--fp16', action='store_false', help='use fp16 (on by default)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of steps to accumulate before backward/update pass.')
    # lr scheduler
    parser.add_argument('--sched', default='cosine', type=str,
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--t_in_epochs', action='store_true',
                        help='update per epoch (instead of per iter)')
    parser.add_argument('--lr_noise', type=float, nargs='+', default=None,
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr_noise_pct', type=float, default=0.67,
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr_noise_std', type=float, default=1.0,
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6,
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay_epochs', type=float, default=30,
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup_steps', type=int, default=500,
                        help='steps to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_epochs', type=int, default=2,
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown_epochs', type=int, default=5,
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience_epochs', type=int, default=10,
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay_rate', '--dr', type=float, default=0.1,
                        help='LR decay rate (default: 0.1)')
    return parser


def add_model_args(parser):
    # models in general
    parser.add_argument('--model_name', type=str, default='resnet18')  # , choices=MODELS)
    parser.add_argument('--try_fused_attn', action='store_false',
                        help='by def uses pytorch fused attn if available (flashattn)')

    # models that do cropping such as tgda or cal
    parser.add_argument('--pre_resize_factor', type=int, default=None,
                        help='factor by which to increase original image size (2: 224 > 448)')

    # knowledge distillation
    parser.add_argument('--legacy_swin_teacher', action='store_false',
                        help='''swin teacher from timm 0.6.13 instead of
                        current (0.9.12), true by def, 
                        use flag if teacher was trained on recent timm''')
    parser.add_argument('--model_name_teacher', type=str, default=None)  # , choices=MODELS)
    parser.add_argument('--teacher_eval_mode', action='store_true')
    parser.add_argument('--temp', type=float, default=0.7)
    parser.add_argument('--loss_orig_weight', type=float, default=1.0)
    parser.add_argument('--loss_kd_weight', type=float, default=1.0)
    parser.add_argument('--student_image_size', type=int, default=None)

    # ablations for tgda
    parser.add_argument('--train_both', action='store_true',
                        help='train teacher with ce for orig and aug while train student')
    parser.add_argument('--teacher_loss', type=str, default=None,
                        choices=[None, 'ce', 'cal'])
    parser.add_argument('--loss_teacher_weight', type=float, default=1.0)
    parser.add_argument('--num_aug_maps', type=int, default=None,
                        help='restrict number of maps for attention')
    parser.add_argument('--cam_aug', action='store_true',
                        help='use CAM for augs instead of attention (conv) module')

    parser.add_argument('--pretrained', action='store_true', help='pretrained model on imagenet')
    parser.add_argument('--ckpt_path', type=str, default=None, help='path to custom pretrained ckpt')
    parser.add_argument('--ckpt_path_teacher', type=str, default=None, help='path to custom pretrained ckpt')
    parser.add_argument('--transfer_learning', action='store_true',
                        help='not load fc layer when using custom ckpt')
    parser.add_argument('--transfer_learning_cal', action='store_true',
                        help='remove the encoder. part (CAL) from the model name')
    parser.add_argument('--convert_cal_student_keep_head', action='store_true',
                        help='same as transfer_learning_cal but keeps head')

    parser.add_argument('--selector', type=str, default=None, choices=[None, 'cal'])
    parser.add_argument('--cal_ap_only', action='store_true', help='cal with no crops at inference')
    parser.add_argument('--tgda', action='store_true', help='teacher guided data augmentation')

    parser.add_argument('--kd_aux_loss', type=str, default=None,
                        choices=['cekd', 'dkd', 'rkd', 'pkt', 'hint',
                                 'trl', 'kl_trl', 'crd', 'std_focal'])
    parser.add_argument('--loss_kd_aux_weight', type=float, default=1.0)
    parser.add_argument('--cont_loss', action='store_true')
    parser.add_argument('--supcon', action='store_true')
    parser.add_argument('--cont_temp', type=float, default=0.07)
    parser.add_argument('--cont_base_temp', type=float, default=None)
    parser.add_argument('--cont_negatives', type=int, default=4096,
                        help='how many negatives if using ContrastMemory (CRD)')
    parser.add_argument('--cont_norm_ind', action='store_true', help='norm for cont loss')
    parser.add_argument('--loss_cont_weight', type=float, default=1.0)

    # focal modulation for supcon
    parser.add_argument('--cont_focal_modulation', type=str, default=None,
                        choices=[None, 'teacher', 'student', 'student_teacher'])
    parser.add_argument('--cont_focal_detach', action='store_true')
    parser.add_argument('--focal_modulation', type=str, default='teacher',
                        choices=[None, 'teacher', 'student', 'student_teacher'])
    parser.add_argument('--modulation_teacher_labels', type=int, default=0)
    parser.add_argument('--modulation_augs', action='store_true')
    parser.add_argument('--cont_focal_gamma', type=float, default=2.0)
    parser.add_argument('--cont_focal_alpha', type=float, default=None)

    parser.add_argument('--layer_names', type=str, nargs='+', default=None)
    parser.add_argument('--num_layers', type=int, default=1)

    parser.add_argument('--layer_names_student', type=str, nargs='+', default=None)
    parser.add_argument('--num_layers_student', type=int, default=1)
 
    parser.add_argument('--pooling_function', type=str, default='gap',
                        choices=['gap', 'bapto', 'abap', 'bap', 'aadgmp', 'adgmp', 'aadgap', 'adgap', 'agap_mean', 'agap_random'])
    parser.add_argument('--pool_size', type=int, default=5)
    parser.add_argument('--disc_feats_norm', action='store_true')
    parser.add_argument('--disc_feats_sign_sqrt', action='store_true')
 
    parser.add_argument('--if_channels', default=[])
    parser.add_argument('--mlp_ratio', type=int, default=0.25)

    # lrresnet
    parser.add_argument('--pos_embedding_type', type=str, default='sin2d',
                        help='positional embedding for encoder, def: learned')

    return parser


def add_augmentation_args(parser):
    # cropping
    parser.add_argument('--resize_size', type=int, default=None, help='resize_size before cropping')
    parser.add_argument('--test_resize_size', type=int, default=None, help='test resize_size before cropping')
    parser.add_argument('--test_square_resize_center_crop', action='store_true',
                        help='resize image to a square of side resize_size then center crop')
    parser.add_argument('--test_resize_directly', action='store_true',
                        help='resizes directly to target image size instead of to larger then center crop')
    parser.add_argument('--random_resized_crop', action='store_true',
                        help='crop random aspect ratio then resize to square')
    parser.add_argument('--square_resize_random_crop', action='store_true',
                        help='resize first to square then random crop')
    parser.add_argument('--short_side_resize_random_crop', action='store_true',
                        help='resize first so short side is resize_size then random crop a square')
    parser.add_argument('--square_resize_center_crop', action='store_true',
                        help='resize first to square then center crop when training')
    # https://github.com/ArdhenduBehera/cap/blob/main/image_datagenerator.py
    parser.add_argument('--affine', action='store_true',
                        help='affine transform as in CAP')
    # flips
    parser.add_argument('--horizontal_flip', action='store_true',
                        help='use horizontal flip when training (on by default)')
    parser.add_argument('--vertical_flip', action='store_true',
                        help='use vertical flip (off by default)')
    # augmentation policies
    parser.add_argument('--auto_aug', action='store_true', help='Auto augmentation used')
    parser.add_argument('--rand_aug', action='store_true', help='RandAugment augmentation used')
    parser.add_argument('--trivial_aug', action='store_true', help='use trivialaugmentwide')
    # color and distortion
    parser.add_argument('--jitter_prob', type=float, default=0.0,
                        help='color jitter probability of applying (0.8 for simclr)')
    parser.add_argument('--jitter_bcs', type=float, default=0.4,
                        help='color jitter brightness contrast saturation (0.4 for simclr)')
    parser.add_argument('--jitter_hue', type=float, default=0.1,
                        help='color jitter hue value (0.1 for simclr)')
    parser.add_argument('--blur', type=float, default=0.0,
                        help='gaussian blur probability (0.5 for simclr)')
    parser.add_argument('--greyscale', type=float, default=0.0,
                        help='gaussian blur probability (0.2 for simclr)')
    parser.add_argument('--solarize_prob', type=float, default=0.0,
                        help='solarize transform probability (0.2 for byol if image_size>32)')
    parser.add_argument('--solarize', type=int, default=128,
                        help='solarize pixels with higher value than (def: 128)')
    # cutmix, mixup, random erasing
    parser.add_argument('--cm', action='store_true', help='Use Cutmix')
    parser.add_argument('--beta', default=1.0, type=float,
                        help='hyperparameter beta (default: 1)')
    parser.add_argument('--mu', action='store_true', help='Use Mixup')
    parser.add_argument('--alpha', default=1.0, type=float,
                        help='mixup interpolation coefficient (default: 1)')
    parser.add_argument('--mix_prob', default=0.5, type=float,
                        help='mixup probability')
    parser.add_argument('--re', default=0.0, type=float,
                        help='Random Erasing probability (def: 0.25)')
    parser.add_argument('--re_sh', default=0.4, type=float, help='max erasing area')
    parser.add_argument('--re_r1', default=0.3, type=float, help='aspect of erasing area')
    # regularization
    parser.add_argument('--ra', type=int, default=0, help='repeated augmentation (def: 3)')
    parser.add_argument('--sd', default=0.0, type=float,
                        help='rate of stochastic depth (def: 0.1)')
    parser.add_argument('--ls', action='store_true', help='label smoothing')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--focal_gamma', type=float, default=0.0,
                        help='gamma in focal loss (def: 2.0)')
    parser.add_argument('--focal_alpha', type=float, default=None,
                        help='alpha in focal loss')
    return parser


def parse_train_args(ret_parser=False):
    parser = add_common_args()

    parser.add_argument('--set_seed_all_before_train_loop', action='store_false',
                        help='if true sets random seed (not np) again just before train loop')
    parser.add_argument('--save_images_wandb', action='store_true')
    parser.add_argument('--save_images', type=int, default=10000,
                        help='save images every x iterations')
    parser.add_argument('--train_trainval', action='store_false',
                        help='when true uses trainval for train and evaluates on test \
                        otherwise use train for train and evaluates on val')

    # test and evaluation modes
    parser.add_argument('--test_only', action='store_true',
                        help='when true skips training and evaluates model on test dataloader')
    parser.add_argument('--test_only_bs1', action='store_true',
                        help='same as test_only but forces bs 1 to emulate streaming/on-demand classification')
    parser.add_argument('--offline', action='store_true',
                        help='do not upload results to wandb')
    parser.add_argument('--test_multiple', type=int, default=4,
                        help='test multiple loops (to reduce model loading time influence)')
    parser.add_argument('--vis_errors', action='store_true',
                        help='when true shows prediction errors (turns on test_only by def)')
    parser.add_argument('--vis_errors_save', action='store_true',
                        help='when true saves prediction errors (turns on test_only by def)')

    # evaluation and metrics
    parser.add_argument('--eval_freq', type=int, default=10, help='eval every x epochs')
    parser.add_argument('--top_k', type=int, default=5, help='for top-k acc metric')

    parser.add_argument('--bottom_k_acc', type=int, nargs='+',
                        default=[1, 5, 10, 25], help='bottom classes accuracy mean')
    parser.add_argument('--bottom_k_acc_percent', action='store_false',
                        help='if true then uses percent of classes rather than an int value')

    # teacher metrics
    parser.add_argument('--compute_train_wise_teacher_metrics', action='store_true',
                    help='if true then computes metrics such as secondary_std, ratios for teacher')
    #https://ieeexplore.ieee.org/document/9839519
    parser.add_argument('--top_k_std', type=int, default=3,
                        help='get the top-k std of predictions for an iteration')
    parser.add_argument('--store_teacher_top_k', action='store_true',
                        help='store teacher logits as csv')
    parser.add_argument('--store_top_k', type=int, default=3,
                        help='store how much top k teacher logits')

    # distributed
    parser.add_argument('--dist_eval', action='store_true',
                        help='validate using dist sampler (else do it on one gpu)')
    parser = add_data_args(parser)
    parser = add_optim_scheduler_args(parser)
    parser = add_augmentation_args(parser)
    parser = add_model_args(parser)
    parser.add_argument("--cfg", type=str, default='configs/soylocal_weakaugs.yaml',
                        help="If using it overwrites args and reads yaml file in given path")

    if ret_parser:
        return parser

    args = parser.parse_args()
    adjust_config(args)
    args = add_adjust_common_dependent(args)

    return args


def parse_inference_args():
    parser = parse_train_args(ret_parser=True)
    parser.add_argument('--images_path', type=str, default='samples',
                        help='path to folder (with images) or image')
    parser.add_argument('--results_inference', type=str, default='results_inference',
                        help='path to folder to save result crops')

    # visualization
    parser.add_argument('--save_crops_only', action='store_false',
                        help='save only crop')
    parser.add_argument('--vis_class', type=int, default=None)
    parser.add_argument('--vis_wrong_only', action='store_true')
    parser.add_argument('--vis_pred_text', action='store_true')
    parser.add_argument('--vis_cols', type=int, default=None,
                        help='how many columns when visualizing images')
    parser.add_argument('--vis_dataset_all', action='store_true',
                        help='if true then visualizes whole dataset otherwise first batch')
    parser.add_argument('--vis_mask_color', action='store_false',
                        help='if true then uses color heat map otherwise attn (white/black) map')
    parser.add_argument('--vis_th_topk', action='store_true',
                        help='for heatmaps use threshold of top-k largest')
    parser.add_argument('--vis_mask_pow', action='store_true',
                        help='square masks when applying heatmap')
    parser.add_argument('--vis_mask_sq', action='store_true',
                        help='square masks when applying heatmap')
    parser.add_argument('--vis_mask', type=str, default=None,
                        help='which layer/mechanism to visualize: gradcam')
    parser.add_argument('--vis_mask_all', action='store_true',
                        help='if true then visualizes all choices in vis_list')
    parser.add_argument('--vis_mask_list', type=str, nargs='+', default=VIS_LIST,
                        help='visualize a few selected methods')
    parser.add_argument('--vis_reg_reduction', type=str, default='mean',
                        help='when visualizing attention from registers')
    # feature metrics
    parser.add_argument('--compute_attention_average', action='store_true',
                        help='otherwise by def computes cka/l2/distances')
    parser.add_argument('--compute_attention_cka', action='store_true',
                        help='otherwise by def computes norm2 output features')
    parser.add_argument('--cka_all_layers', action='store_true',
                        help='if used then uses all layers of vit for cka analysis')
    parser.add_argument('--compute_batch_norm_stats', action='store_true')
    parser.add_argument('--weights_type', type=str, default=None,
                        choices=[None, 'norm', 'main'])

    args = parser.parse_args()
    adjust_config(args)
    args = add_adjust_common_dependent(args)
    return args


def adjust_config(args):
    if args.cfg:
        config = yaml_config_hook(os.path.abspath(args.cfg))
        for k, v in config.items():
            if hasattr(args, k):
                setattr(args, k, v)


if __name__ == '__main__':
    args = parse_train_args()
    print(args)
