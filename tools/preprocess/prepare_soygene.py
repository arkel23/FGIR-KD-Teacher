import os
import argparse
import pandas as pd


def save_df_train_test(args):
    anno_train_fp = os.path.join(args.root_path, 'anno', 'train.txt')

    df_train = pd.read_csv(anno_train_fp, names=['dir', 'class_id'], sep=' ')
    df_train['class_id'] = df_train['class_id'] - 1

    save_fp = os.path.join(args.root_path, 'train_val.csv')
    df_train.to_csv(save_fp, sep=',', header=True, index=False, columns=['class_id', 'dir'])
    print('Train set: ', len(df_train))

    anno_test_fp = os.path.join(args.root_path, 'anno', 'test.txt')

    df_test = pd.read_csv(anno_test_fp, names=['dir', 'class_id'], sep=' ')
    df_test['class_id'] = df_test['class_id'] - 1

    save_fp = os.path.join(args.root_path, 'test.csv')
    df_test.to_csv(save_fp, sep=',', header=True, index=False, columns=['class_id', 'dir'])
    print('Test set: ', len(df_test))

    classes_total = len(df_test['class_id'].unique())
    print('Number of classes: ', classes_total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='../../data/soygene/',
                        help='path to folder like IN')
    args = parser.parse_args()

    save_df_train_test(args)


main()