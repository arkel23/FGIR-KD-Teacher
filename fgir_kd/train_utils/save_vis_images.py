import sys
import os.path as osp

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image
from einops import rearrange
import wandb


def inverse_normalize(tensor, norm_custom=False):
    if norm_custom:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor


def vis_images(args, curr_img, img, title=None):
    img = rearrange(img, '1 c h w -> h w c')

    img = inverse_normalize(img.data, args.custom_mean_std)
    img = np.uint8(np.clip(img.cpu().numpy(), 0, 1))

    fig = plt.figure()
    axs = fig.add_subplot(111)

    axs.imshow(img)
    if title is not None:
        axs.set_title(title, fontsize=8, wrap=True)

    axs.axis('off')
    fig.tight_layout()

    if args.vis_errors_save:
        fp = osp.join(args.results_dir, f'test_{curr_img}.png')
        fig.savefig(fp, dpi=300)
    else:
        plt.show(block=False)

        inp = input("input 'exit' to stop visualizing: ")
        if inp == 'exit':
            sys.exit()

    plt.close()
    return 0


def save_samples(images, output, train, curr_iter, saved, args):
    if hasattr(args, 'pre_resize_factor') and args.pre_resize_factor:
        img_sz = args.image_size * args.pre_resize_factor
    else:
        img_sz = args.image_size

    if args.selector == 'cal':
        if isinstance(output, tuple) and len(output) == 7:
            _, _, _, _, _, _, crops = output
        elif isinstance(output, tuple) and len(output) == 4 and (not args.cont_loss and not args.kd_aux_loss):
            _, _, _, crops = output
        elif isinstance(output, tuple) and len(output) == 3 and (not args.cont_loss and not args.kd_aux_loss):
            _, _, crops = output
        elif isinstance(output, tuple) and len(output) == 2 and not args.cont_loss:
            _, crops = output
        else:
            crops = None

    else:
        crops = None

    if crops is not None and args.selector == 'cal' and args.student_image_size:
        crops_sz = args.student_image_size    
    else:
        crops_sz = args.image_size


    if args.save_images and train and (curr_iter % args.save_images == 0):
        save_images(images, osp.join(args.results_dir, f'{curr_iter}.png'), img_sz, args.custom_mean_std)
        if crops is not None:
            save_images(crops, osp.join(args.results_dir, f'{curr_iter}_crops.png'), crops_sz, args.custom_mean_std)

    elif args.save_images and not train and not saved:
        save_images(images, osp.join(args.results_dir, f'test.png'), img_sz, args.custom_mean_std)
        if crops is not None:
            save_images(crops, osp.join(args.results_dir, f'test_crops.png'), crops_sz, args.custom_mean_std)

        if not args.debugging and not args.vis_errors and not args.offline:
            if crops is not None and hasattr(args, 'wandb_save_images') and args.wandb_save_images:
                wandb.log({'crops': wandb.Image(crops)})
            elif hasattr(args, 'wandb_save_images') and args.wandb_save_images:
                wandb.log({'crops': wandb.Image(images)})

        return True
    
    if saved:
        return True
    return False


def save_images(images, fp, img_sz, custom_mean_std):
    with torch.no_grad():
        samples = images.reshape(-1, 3, img_sz, img_sz)
        samples = inverse_normalize(images.data, custom_mean_std)
        save_image(samples, fp, nrow=int(np.sqrt(samples.shape[0])))
    return 0
