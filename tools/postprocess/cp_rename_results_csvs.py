import os
import glob
import shutil
import argparse

def rename_based_on_folder_file_pattern(folder, file_pattern, suffix=''):
    name = os.path.splitext(file_pattern)[0]

    fp = os.path.join(folder, '**', file_pattern)
    files_all = glob.glob(fp, recursive=True)

    results_dir = f'{folder}_{name}'
    os.makedirs(results_dir, exist_ok=True)

    for i, file in enumerate(files_all):
        ds_mn_serial = file.split('/')[-2]
        dataset_name = ds_mn_serial.split('_')[0]
        mn_serial = '_'.join(ds_mn_serial.split('_')[1:])

        full_fn = os.path.join(results_dir, f'{dataset_name}_{mn_serial}{suffix}.csv')

        shutil.copyfile(file, full_fn)

        print(f'{i}/{len(files_all)}: {file} copied as {full_fn}')

    return 0

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--folder', type=str, default='ft')
    parser.add_argument('--file_pattern', type=str, default='ind_preds.csv')
    parser.add_argument('--suffix', type=str, default='',
                        help='optional suffix at end of name')

    args = parser.parse_args()

    rename_based_on_folder_file_pattern(args.folder, args.file_pattern, args.suffix)

    return 0

main()


