import os
import argparse
import pandas as pd


def save_df_train_test(args):
    classes_total = 0
    df_train_list = []
    df_test_list = []

    for root, folder in zip(args.dataset_roots, args.folder_images):
        print(root, folder)
        classes_total, df_train, df_test = save_individual_df_train_test(args, root, folder, classes_total)
        df_train_list.append(df_train)
        df_test_list.append(df_test)

    df_train = pd.concat(df_train_list)
    save_fp = os.path.join(args.root_path, f'{args.save_name}_{args.df_train}')
    df_train.to_csv(save_fp, sep=',', header=True, index=False, columns=['class_id', 'dir'])
    print(len(df_train))

    df_test = pd.concat(df_test_list)
    save_fp = os.path.join(args.root_path, f'{args.save_name}_{args.df_test}')
    df_test.to_csv(save_fp, sep=',', header=True, index=False, columns=['class_id', 'dir'])
    print(len(df_test))


def save_individual_df_train_test(args, root, folder, classes_total=0):
    train_fp = os.path.join(args.root_path, root, args.df_train)

    df_train = pd.read_csv(train_fp, sep=',')
    df_train['class_id'] = df_train['class_id'] + classes_total
    df_train['dir'] = df_train['dir'].apply(lambda x: os.path.join(root, folder, x))

    print(len(df_train))

    test_fp = os.path.join(args.root_path, root, args.df_test)

    df_test = pd.read_csv(test_fp, sep=',')
    df_test['class_id'] = df_test['class_id'] + classes_total
    df_test['dir'] = df_test['dir'].apply(lambda x: os.path.join(root, folder, x))

    print(len(df_test))

    print(classes_total)
    classes_total += len(df_test['class_id'].unique())
    print(classes_total)

    return classes_total, df_train, df_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='../../data/',
                        help='path to folder like IN')
    parser.add_argument('--df_train', type=str, default='train_val.csv')
    parser.add_argument('--df_test', type=str, default='test.csv')
    parser.add_argument('--dataset_roots', nargs='+', type=str,
                        default=['aircraft/fgvc-aircraft-2013b/data',
                                 'cars',
                                 'cub/CUB_200_2011']
    )
    parser.add_argument('--folder_images', nargs='+', type=str,
                        default=['images', 'car_ims', 'images'])
    parser.add_argument('--save_name', type=str, default='hierarchical')
    args = parser.parse_args()

    save_df_train_test(args)


main()
