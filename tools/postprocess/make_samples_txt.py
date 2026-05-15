import pandas as pd
import yaml
import os

datasets = ['aircraft', 'cars', 'cub', 'dogs', 'flowers', 'moe', 'pets']
dirs = []

for dataset in datasets:
    with open(f'configs/datasets/{dataset}.yaml') as f:
         cfg = yaml.safe_load(f)
    df = pd.read_csv(f'data/{dataset}/test.csv')
    fp = os.path.join(cfg['dataset_root_path'], cfg['folder_test'], df.iloc[0]['dir'])
    dirs.append(fp)
    fp = os.path.join(cfg['dataset_root_path'], cfg['folder_test'], df.iloc[-1]['dir'])
    dirs.append(fp)

df = pd.DataFrame(dirs, columns=['dir'])
df.to_csv('samples.txt', header=True, index=False)
