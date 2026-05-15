import os
import argparse
import pandas as pd


def save_df_train_val_test(folder):
    # train
    anno_train_fp = os.path.join(folder, 'train800.txt')
    df_train = pd.read_csv(anno_train_fp, names=['dir', 'class_id'], sep=' ')
    save_fp = os.path.join(folder, f'train.csv')
    df_train.to_csv(save_fp, sep=',', header=True, index=False, columns=['class_id', 'dir'])

    # val
    anno_val_fp = os.path.join(folder, 'val200.txt')
    df_val = pd.read_csv(anno_val_fp, names=['dir', 'class_id'], sep=' ')
    save_fp = os.path.join(folder, f'val.csv')
    df_val.to_csv(save_fp, sep=',', header=True, index=False, columns=['class_id', 'dir'])

    # train_val
    anno_train_val_fp = os.path.join(folder, 'train800val200.txt')
    df_train_val = pd.read_csv(anno_train_val_fp, names=['dir', 'class_id'], sep=' ')
    save_fp = os.path.join(folder, f'train_val.csv')
    df_train_val.to_csv(save_fp, sep=',', header=True, index=False, columns=['class_id', 'dir'])

    # test
    anno_test_fp = os.path.join(folder, 'test.txt')
    df_test = pd.read_csv(anno_test_fp, names=['dir', 'class_id'], sep=' ')
    save_fp = os.path.join(folder, f'test.csv')
    df_test.to_csv(save_fp, sep=',', header=True, index=False, columns=['class_id', 'dir'])

    print(f'{folder} test set: ', len(df_test))

    classes_total = len(df_test['class_id'].unique())
    print(f'Number of classes in {folder}: ', classes_total)

    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='../../data/vtab/vtab-1k',
                        help='path to root folder')
    args = parser.parse_args()

    folders = ('caltech101', 'cifar', 'clevr_count', 'clevr_dist', 'diabetic_retinopathy',
               'dmlab', 'dsprites_loc', 'dsprites_ori', 'dtd', 'eurosat', 'kitti',
               'oxford_flowers102', 'oxford_iiit_pet', 'patch_camelyon', 'resisc45',
               'smallnorb_azi', 'smallnorb_ele', 'sun397', 'svhn')

    for folder in folders:
        save_df_train_val_test(os.path.join(args.root_path, folder))

    return 0


main()

