import os
import glob
import shutil
import argparse
import torch


def pop_keys(fp, ema=False, optimizer=False):
    state_dict = torch.load(fp, map_location='cpu')
    if ema:
        state_dict.pop('model_ema', None)
        print(f'model_ema popped from {fp}')
    if optimizer:
        state_dict.pop('optimizer', None)
        print(f'optimizer popped from {fp}')
    return state_dict


def rename_based_on_folder_file_pattern(folder, file_pattern, suffix='',
                                        pop_model_ema=False, pop_optimizer=False):
    fp = os.path.join(folder, '**', file_pattern)
    files_all = glob.glob(fp, recursive=True)

    results_dir = f'{folder}_ckpts'
    os.makedirs(results_dir, exist_ok=True)

    for i, file in enumerate(files_all):
        if pop_model_ema or pop_optimizer:
            state_dict = pop_keys(file, pop_model_ema, pop_optimizer)

        ds_mn_serial = file.split('/')[-2]
        dataset_name = ds_mn_serial.split('_')[0]
        mn_serial = '_'.join(ds_mn_serial.split('_')[1:])

        full_fn = os.path.join(results_dir, f'{dataset_name}_{mn_serial}{suffix}.pth')

        if pop_model_ema or pop_optimizer:
            torch.save(state_dict, full_fn)
        else:
            shutil.copyfile(file, full_fn)

        print(f'{i}/{len(files_all)}: {file} copied as {full_fn}')

    return 0


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--folder', type=str, default='results_train')
    parser.add_argument('--file_pattern', type=str, default='*last.pth')
    parser.add_argument('--suffix', type=str, default='',
                        help='optional suffix at end of name')
    parser.add_argument('--pop_model_ema', action='store_true',
                        help='if true then pops model_ema key if it exists')
    parser.add_argument('--pop_optimizer', action='store_false',
                        help='if true then pops optimizer key if it exists')

    args = parser.parse_args()

    rename_based_on_folder_file_pattern(args.folder, args.file_pattern, args.suffix,
                                        args.pop_model_ema, args.pop_optimizer)

    return 0

main()
