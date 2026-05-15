import os

import torch
from torchvision.utils import save_image

from fgir_kd.data_utils.build_dataloaders import build_dataloaders
from fgir_kd.other_utils.build_args import parse_inference_args
from fgir_kd.train_utils.misc_utils import set_random_seed
from fgir_kd.train_utils.save_vis_images import inverse_normalize


def adjust_args_general(args):
    args.run_name = '{}_{}'.format(args.dataset_name, args.serial)
    args.results_dir = os.path.join(args.results_inference, args.run_name)
    os.makedirs(args.results_dir, exist_ok=True)
    return args


def preprocess_save_images(images, targets, args, loader, split='train', idx=0):
    images = inverse_normalize(images.data, norm_custom=args.custom_mean_std)

    if args.vis_class is not None:
        fp = os.path.join(args.results_dir, f'{split}_{args.vis_class}.png')
    else:
        fp = os.path.join(args.results_dir, f'{split}_{idx}.png')

    number_imgs = images.shape[0]
    ncols = args.vis_cols if args.vis_cols else int(number_imgs ** 0.5)
    # nrow is the number of images per row (so the number of columns)
    save_image(images, fp, nrow=ncols, padding=2)

    if idx % args.log_freq == 0 or args.vis_class is not None:
        print(f'{split} ({idx} / {len(loader)}): {fp}')
        print(targets)

    return 0

def vis_dataset(args):

    set_random_seed(args.seed, numpy=False)

    # dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(args)

    args = adjust_args_general(args)
    # print(args)

    for split, loader in zip(['test', 'train'], [test_loader, train_loader]):
        if args.vis_class is not None:
            images_class = []
            targets_class = []

        for idx, (images, targets) in enumerate(loader):
            if args.vis_class is None:
                preprocess_save_images(images, targets, args, loader, split, idx)

                if not args.vis_dataset_all:
                    break
                else:
                    continue

            for image, target in zip(images, targets):
                if target == args.vis_class:
                    images_class.append(image)
                    targets_class.append(target)

            if len(images_class) >= args.batch_size or idx == len(loader) - 1:
                images_class = torch.stack(images_class[:args.batch_size], dim=0)
                targets_class = torch.stack(targets_class[:args.batch_size], dim=0)
                preprocess_save_images(images_class, targets_class, args, loader, split, idx)
                break
            else:
                continue

    return 0


def main():
    args = parse_inference_args()

    vis_dataset(args)

    return 0


if __name__ == '__main__':
    main()