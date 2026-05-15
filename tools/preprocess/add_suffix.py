import os
import argparse
import pandas as pd

def add_suffix(args):
    df = args.df_input

    name = df # name of df
    df = pd.read_csv(df) # read df
    suffix = args.suffix # suffix to add
    column = args.column # column to manipulate

    # add suffix to specifed column
    print('df before adding suffix: \n', df.head)
    df[column] = suffix + df[column]
    print('df after adding suffix: \n', df.head)

    # save file
    if args.save_name:
        save = args.save_name + '_' + name
        df.to_csv(save, index=False)
    else:
        df.to_csv(name, index=False)

    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', type=str,
                        help='suffix path to add')
    parser.add_argument('--column', type=str, default='dir',
                        help='column to add suffix')
    parser.add_argument('--df_input', type=str, default='train_val.csv',
                        help='input dataframe, e.g train_val.csv')
    parser.add_argument('--save_name', type=str)
    args = parser.parse_args()

    add_suffix(args)


main()