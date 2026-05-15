# https://github.com/aanna0701/SPT_LSA_ViT/blob/4446cb5aad3ae94acfed9e517ba36d7988dfc0a1/utils/mix.py#L1
import numpy as np
import torch


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def mixup_data(x, y, args):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if args.alpha > 0:
        lam = np.random.beta(args.alpha, args.alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    device = x.device

    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, args):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if args.beta > 0:
        lam = np.random.beta(args.beta, args.beta)
    else:
        lam = 1

    batch_size = x.size()[0]
    device = x.device

    index = torch.randperm(batch_size).to(device)

    y_a, y_b = y, y[index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x_sliced = x[index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    return [bbx1, bby1, bbx2, bby2], y_a, y_b, lam, x_sliced


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def prepare_mix(images, targets, args):

    # cutmix and mixup
    if args.cm and args.mu:
        switching_prob = np.random.rand(1)
        # Cutmix
        if switching_prob < 0.5:
            slicing_idx, y_a, y_b, lam, sliced = cutmix_data(images, targets, args)
            images[:, :, slicing_idx[0]:slicing_idx[2], slicing_idx[1]:slicing_idx[3]] = sliced
        # Mixup
        else:
            images, y_a, y_b, lam = mixup_data(images, targets, args)
    # cutmix only
    elif args.cm:
        slicing_idx, y_a, y_b, lam, sliced = cutmix_data(images, targets, args)
        images[:, :, slicing_idx[0]:slicing_idx[2], slicing_idx[1]:slicing_idx[3]] = sliced
    # mixup only
    elif args.mu:
        images, y_a, y_b, lam = mixup_data(images, targets, args)
    return images, y_a, y_b, lam


def get_mix(images, targets, train, args):
    y_a, y_b, lam = None, None, None

    if args.cm or args.mu:
        r = np.random.rand(1)
        if r < args.mix_prob and train:
            images, y_a, y_b, lam = prepare_mix(images, targets, args)

    return images, y_a, y_b, lam
