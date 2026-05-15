import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFile

import torch
import torch.utils.data as data
from torchvision import datasets

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except:
    from PIL.Image import BICUBIC as BICUBIC

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_set(args, split, transform=None):
    if args.dataset_name == 'cifar10':
        ds = datasets.CIFAR10(root=args.dataset_root_path,
                              train=True if split == 'train' else False,
                              transform=transform, download=True)
        ds.num_classes = 10
    elif args.dataset_name == 'cifar100':
        ds = datasets.CIFAR100(root=args.dataset_root_path,
                               train=True if split == 'train' else False,
                               transform=transform, download=True)
        ds.num_classes = 100
    else:
        if args.kd_aux_loss == 'crd' and split == 'train':
            ds = DatasetImgTargetContIDX(args, split=split, transform=transform)
        elif args.dataset_in_memory:
            ds = DatasetImgTargetInMemory(args, split=split, transform=transform)
        else:
            ds = DatasetImgTarget(args, split=split, transform=transform)
        args.num_classes = ds.num_classes

    setattr(args, f'num_images_{split}', ds.__len__())
    print(f"{args.dataset_name} {split} split. N={ds.__len__()}, K={ds.num_classes}.")
    return ds


class DatasetImgTarget(data.Dataset):
    def __init__(self, args, split, transform=None):
        self.root = os.path.abspath(args.dataset_root_path)
        self.transform = transform
        self.dataset_name = args.dataset_name

        if split == 'train':
            if args.train_trainval:
                self.images_folder = args.folder_train
                self.df_file_name = args.df_trainval
            else:
                self.images_folder = args.folder_train
                self.df_file_name = args.df_train
        elif split == 'val':
            if args.train_trainval:
                self.images_folder = args.folder_test
                self.df_file_name = args.df_test
            else:
                self.images_folder = args.folder_val
                self.df_file_name = args.df_val
        else:
            self.images_folder = args.folder_test
            self.df_file_name = args.df_test

        assert os.path.isfile(os.path.join(self.root, self.df_file_name)), \
            f'{os.path.join(self.root, self.df_file_name)} is not a file.'

        self.df = pd.read_csv(os.path.join(self.root, self.df_file_name), sep=',')
        self.targets = self.df['class_id'].to_numpy()
        self.data = self.df['dir'].to_numpy()

        self.num_classes = len(np.unique(self.targets))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_dir, target = self.data[idx], self.targets[idx]
        full_img_dir = os.path.join(self.root, self.images_folder, img_dir)
        img = Image.open(full_img_dir)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.targets)


class DatasetImgTargetInMemory(DatasetImgTarget):
    def __init__(self, args, split, transform=None):
        super().__init__(args, split, transform)

        self.images = [None] * len(self.data)

        def load_image(i, img_dir):
        # for i, img_dir in enumerate(self.data):
            full_img_dir = os.path.join(self.root, self.images_folder, img_dir)
            img = Image.open(full_img_dir)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((args.resize_size, args.resize_size))
            self.images[i] = img

        max_workers = args.cpu_workers if args.cpu_workers else 8
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(load_image, i, img_dir) for i, img_dir in enumerate(self.data)]
            for f in futures:
                f.result()  # Wait for all threads to complete

        print(f'Loaded all {split} images on memory')

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img, target = self.images[idx], self.targets[idx]

        if self.transform:
            img = self.transform(img)

        return img, target


class DatasetImgTargetContIDX(data.Dataset):
    def __init__(self, args, split, transform=None, percent=None):
        self.root = os.path.abspath(args.dataset_root_path)
        self.transform = transform

        if split == 'train':
            if args.train_trainval:
                self.images_folder = args.folder_train
                self.df_file_name = args.df_trainval
            else:
                self.images_folder = args.folder_train
                self.df_file_name = args.df_train
        elif split == 'val':
            if args.train_trainval:
                self.images_folder = args.folder_test
                self.df_file_name = args.df_test
            else:
                self.images_folder = args.folder_val
                self.df_file_name = args.df_val
        else:
            self.images_folder = args.folder_test
            self.df_file_name = args.df_test

        assert os.path.isfile(os.path.join(self.root, self.df_file_name)), \
            f'{os.path.join(self.root, self.df_file_name)} is not a file.'

        self.df = pd.read_csv(os.path.join(self.root, self.df_file_name), sep=',')
        self.targets = self.df['class_id'].to_numpy()
        self.data = self.df['dir'].to_numpy()

        self.num_classes = len(np.unique(self.targets))

        # contrastive index hparams
        # number of contrastive negative pairs
        self.k = getattr(args, 'cont_k', 4096)
        self.mode = getattr(args, 'cont_mode', 'exact')

        # positives: list with indexes corresponding to each class
        self.cls_positive = [[] for i in range(self.num_classes)]
        for i in range(len(self.targets)):
            self.cls_positive[self.targets[i]].append(i)

        # negatives: list with indexes that do not belong in each class
        self.cls_negative = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(self.num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(self.num_classes)]

        if percent is not None and (0 < percent < 1):
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(self.num_classes)]

        # depending on the numpy version this may give an error since it will
        # try to create a 2d array, adding dtype=object keeps it separate
        try:
            self.cls_positive = np.asarray(self.cls_positive)
            self.cls_negative = np.asarray(self.cls_negative)
        except:
            self.cls_positive = np.asarray(self.cls_positive, dtype="object")
            self.cls_negative = np.asarray(self.cls_negative, dtype="object")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_dir, target = self.data[idx], self.targets[idx]
        full_img_dir = os.path.join(self.root, self.images_folder, img_dir)
        img = Image.open(full_img_dir)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = idx
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))

        return img, target, idx, sample_idx

    def __len__(self):
        return len(self.targets)
