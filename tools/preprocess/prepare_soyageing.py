import os
import argparse
import pandas as pd


def save_df_train_test(args):
    df_train_list = []
    df_test_list = []

    for folder in args.folders:
        df_train, df_test = save_individual_df_train_test(args, folder)
        df_train_list.append(df_train)
        df_test_list.append(df_test)

    df_train = pd.concat(df_train_list)
    save_fp = os.path.join(args.root_path, f'train_val.csv')
    df_train.to_csv(save_fp, sep=',', header=True, index=False, columns=['class_id', 'dir'])
    print('Aggregated train set: ', len(df_train))

    df_test = pd.concat(df_test_list)
    save_fp = os.path.join(args.root_path, f'test.csv')
    df_test.to_csv(save_fp, sep=',', header=True, index=False, columns=['class_id', 'dir'])
    print('Aggregated test set: ', len(df_test))

    classes_total = len(df_test['class_id'].unique())
    print('Total number of classes: ', classes_total)


def save_individual_df_train_test(args, folder):
    anno_train_fp = os.path.join(args.root_path, folder, 'anno', 'train.txt')

    df_train = pd.read_csv(anno_train_fp, names=['dir', 'class_id'], sep=' ')
    df_train['class_id'] = df_train['class_id'] - 1
    df_train['dir'] = df_train['dir'].apply(lambda x: os.path.join(folder, 'images', x))

    print(f'{folder} train set: ', len(df_train))

    anno_test_fp = os.path.join(args.root_path, folder, 'anno', 'test.txt')

    df_test = pd.read_csv(anno_test_fp, names=['dir', 'class_id'], sep=' ')
    df_test['class_id'] = df_test['class_id'] - 1
    df_test['dir'] = df_test['dir'].apply(lambda x: os.path.join(folder, 'images', x))

    print(f'{folder} test set: ', len(df_test))

    # save individual subsets
    save_fp = os.path.join(args.root_path, f'train_val_{folder}.csv')
    df_train.to_csv(save_fp, sep=',', header=True, index=False, columns=['class_id', 'dir'])
    print(f'Individual train set {folder}: ', len(df_train))

    save_fp = os.path.join(args.root_path, f'test_{folder}.csv')
    df_test.to_csv(save_fp, sep=',', header=True, index=False, columns=['class_id', 'dir'])
    print(f'Individual test set {folder}: ', len(df_test))

    classes_total = len(df_test['class_id'].unique())
    print('Number of classes in {folder}: ', classes_total)

    return df_train, df_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='../../data/soyageing/',
                        help='path to root folder')
    parser.add_argument('--folders', nargs='+', type=str,
                        default=['R1', 'R3', 'R4', 'R5', 'R6'])
    args = parser.parse_args()

    save_df_train_test(args)


main()
