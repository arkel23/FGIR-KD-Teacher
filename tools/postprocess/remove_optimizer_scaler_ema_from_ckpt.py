import os
import glob
import argparse
import torch


def pop_keys(fp, ema=False, scaler=False, optimizer=False):
    state_dict = torch.load(fp, map_location='cpu')
    if ema:
        state_dict.pop('model_ema', None)
        print(f'model_ema popped from {fp}')
    if scaler:
        state_dict.pop('scaler', None)
        print(f'scaler popped from {fp}')
    if optimizer:
        state_dict.pop('optimizer', None)
        print(f'optimizer popped from {fp}')
    return state_dict


def rename_based_on_folder_file_pattern(
    folder, file_pattern, suffix='',
    pop_model_ema=False, pop_scaler=False, pop_optimizer=False):

    fp = os.path.join(folder, '**', file_pattern)
    files_all = glob.glob(fp, recursive=True)

    print(fp, len(files_all))

    for i, file in enumerate(files_all):
        if pop_model_ema or pop_optimizer:
          state_dict = pop_keys(file, pop_model_ema, pop_scaler, pop_optimizer)

        if pop_model_ema or pop_optimizer:
           torch.save(state_dict, file)

        print(f'{i}/{len(files_all)}: {file} modified as {file}')

    return 0


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--folder', type=str, default='results_train')
    parser.add_argument('--file_pattern', type=str, default='*.pth')
    parser.add_argument('--suffix', type=str, default='',
                        help='optional suffix at end of name')
    parser.add_argument('--pop_model_ema', action='store_false',
                        help='if true then pops model_ema key if it exists')
    parser.add_argument('--pop_scaler', action='store_false',
                        help='if true then pops scaler key if it exists')
    parser.add_argument('--pop_optimizer', action='store_false',
                        help='if true then pops optimizer key if it exists')

    args = parser.parse_args()

    rename_based_on_folder_file_pattern(
        args.folder, args.file_pattern, args.suffix,
        args.pop_model_ema, args.pop_scaler, args.pop_optimizer)

    return 0

main()

