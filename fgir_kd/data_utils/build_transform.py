import torch
from torchvision import transforms
import torchvision.transforms.functional as F

from .augmentations import CIFAR10Policy, SVHNPolicy, ImageNetPolicy

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except:
    from PIL.Image import BICUBIC as BICUBIC


MEANS = {
    '05': (0.5, 0.5, 0.5),
    'imagenet': (0.485, 0.456, 0.406),
    'tinyin': (0.4802, 0.4481, 0.3975),
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5070, 0.4865, 0.4409),
    'svhn': (0.4377, 0.4438, 0.4728),
    'cub': (0.3659524, 0.42010019, 0.41562049)
}

STDS = {
    '05': (0.5, 0.5, 0.5),
    'imagenet': (0.229, 0.224, 0.225),
    'tinyin': (0.2770, 0.2691, 0.2821),
    'cifar10': (0.2470, 0.2435, 0.2616),
    'cifar100': (0.2673, 0.2564, 0.2762),
    'svhn': (0.1980, 0.2010, 0.1970),
    'cub': (0.07625843, 0.04599726, 0.06182727)
}



class ResizeAndPad:
    def __init__(self, image_size, padding_value=0):
        self.image_size = image_size
        self.padding_value = padding_value

    def __call__(self, img):
        # Resize the image so that the long side is image_size
        w, h = img.size
        if w > h:
            new_w = self.image_size
            new_h = int(self.image_size * h / w)
        else:
            new_h = self.image_size
            new_w = int(self.image_size * w / h)
        
        img = F.resize(img, (new_h, new_w))
        
        # Calculate padding
        pad_h = self.image_size - new_h
        pad_w = self.image_size - new_w
        
        # Padding on all sides
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        
        img = F.pad(img, padding, fill=self.padding_value)
        
        return img


def standard_transform(args, is_train):
    if hasattr(args, 'pre_resize_factor') and args.pre_resize_factor:
        image_size = args.image_size * args.pre_resize_factor
        resize_size = args.resize_size * args.pre_resize_factor
        test_resize_size = args.test_resize_size * args.pre_resize_factor
    else:
        image_size = args.image_size
        resize_size = args.resize_size
        test_resize_size = args.test_resize_size

    mean = MEANS['imagenet']
    std = STDS['imagenet']
    if args.custom_mean_std:
        mean = MEANS[args.dataset_name] if args.dataset_name in MEANS.keys() else MEANS['05']
        std = STDS[args.dataset_name] if args.dataset_name in STDS.keys() else STDS['05']

    if (args.dataset_name =='cifar10' or args.dataset_name == 'cifar100') and image_size == 32:
        aa = CIFAR10Policy()
    elif args.dataset_name == 'svhn' and image_size == 32:
        aa = SVHNPolicy()
    else:
        aa = ImageNetPolicy()

    t = []

    if is_train:
        if args.affine:
            t.append(transforms.Resize(
                (resize_size, resize_size), interpolation=BICUBIC))
            t.append(transforms.RandomAffine(degrees=15, scale=(0.85, 1.15),
                                             interpolation=BICUBIC))
            t.append(transforms.RandomCrop((image_size, image_size)))
        elif args.random_resized_crop:
            t.append(transforms.RandomResizedCrop(
                image_size, interpolation=BICUBIC))
        elif args.square_resize_random_crop:
            t.append(transforms.Resize(
                (resize_size, resize_size),
                interpolation=BICUBIC))
            t.append(transforms.RandomCrop(image_size))
        elif args.short_side_resize_random_crop:
            t.append(transforms.Resize(
                resize_size, interpolation=BICUBIC))
            t.append(transforms.RandomCrop((image_size, image_size)))
        elif args.square_resize_center_crop:
            t.append(transforms.Resize(
                (resize_size, resize_size),
                interpolation=BICUBIC))
            t.append(transforms.CenterCrop(image_size))
        else:
            t.append(ResizeAndPad(image_size, padding_value=255)) 

        if args.horizontal_flip:
            t.append(transforms.RandomHorizontalFlip())
        if args.vertical_flip:
            t.append(transforms.RandomVerticalFlip())
        if args.jitter_prob > 0:
            t.append(transforms.RandomApply([transforms.ColorJitter(
                brightness=args.jitter_bcs, contrast=args.jitter_bcs,
                saturation=args.jitter_bcs, hue=args.jitter_hue)], p=args.jitter_prob))
        if args.greyscale > 0:
            t.append(transforms.RandomGrayscale(p=args.greyscale))
        if args.blur > 0:
            t.append(transforms.RandomApply(
                [transforms.GaussianBlur(
                    kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=args.blur))
        if args.solarize_prob > 0:
            t.append(transforms.RandomApply(
                [transforms.RandomSolarize(args.solarize, p=args.solarize_prob)]))
        if args.auto_aug:
            t.append(aa)
        if args.rand_aug:
            t.append(transforms.RandAugment())
        if args.trivial_aug:
            t.append(transforms.TrivialAugmentWide())
    else:
        if ((args.dataset_name in ['cifar10', 'cifar100', 'svhn'] and image_size == 32)
           or (args.dataset_name == 'tinyin' and image_size == 64)):
            t.append(transforms.Resize(image_size))
        else:
            if args.test_resize_directly:
                t.append(transforms.Resize(
                    (image_size, image_size),
                    interpolation=BICUBIC))
            elif args.test_square_resize_center_crop:
                t.append(transforms.Resize(
                    (test_resize_size, test_resize_size),
                    interpolation=BICUBIC))
                t.append(transforms.CenterCrop(image_size))
            else:
                t.append(ResizeAndPad(image_size, padding_value=255))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean=mean, std=std))
    if is_train and args.re > 0:
        t.append(transforms.RandomErasing(
            p=args.re, scale=(0.02, args.re_sh), ratio=(args.re_r1, 3.3)))
    transform = transforms.Compose(t)
    print(transform)
    return transform


def build_transform(args, split):
    is_train = True if split == 'train' else False

    transform = standard_transform(args, is_train)

    return transform
