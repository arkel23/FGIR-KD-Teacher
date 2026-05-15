import os
import yaml
import argparse
import pandas as pd


def yaml_config_hook(config_file):
    """
    Custom YAML config loader, which can include other yaml files (I like using config files
    insteaad of using argparser)
    """

    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            fp = cfg.get("defaults").get(d)
            cf = os.path.join(os.path.dirname(config_file), fp)
            with open(cf) as f:
                val = yaml.safe_load(f)
                print(val)
                cfg.update(val)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg


def update_ind_preds(args):
    # read ind preds, val/test.csv, and classid_classname.csv files
    ind_preds = pd.read_csv(args.preds_path)

    fp = os.path.join(args.dataset_root_path, args.df_classid_classname)
    dic_classid_classname = pd.read_csv(fp, index_col='class_id')['class_name'].to_dict()

    fn_test = args.df_test if args.train_trainval else args.df_val
    folder_test = args.folder_test if args.train_trainval else args.folder_val
    fp = os.path.join(args.dataset_root_path, fn_test)
    df_test = pd.read_csv(fp)

    print(ind_preds.head())

    # keep all ind_preds or filter by only wrong
    if args.save_wrong_preds_only:
        print('Before filtering wrong only: ', len(ind_preds))

        ind_preds = ind_preds[ind_preds['class_id'] != ind_preds['pred_id']]

        print('After filtering wrong only: ', len(ind_preds))

    # filter by confidently wrong
    if args.save_prob_th:
        print('Before filtering with confidence threshold: ', len(ind_preds))

        ind_preds = ind_preds[ind_preds['prob'] >= args.save_prob_th]

        print('After filtering with confidence threshold: ', len(ind_preds))

    # update ind_preds with class_name and pred_class_name columns
    ind_preds['class_name'] = ind_preds['class_id'].apply(lambda x: dic_classid_classname[x])
    ind_preds['pred_class_name'] = ind_preds['pred_id'].apply(lambda x: dic_classid_classname[x])

    # update with full dir
    ind_preds['dir'] = df_test['dir'].apply(lambda x: os.path.join(args.dataset_root_path, folder_test, x))

    # save only certain columns
    df_updated = ind_preds[['dir', 'class_name', 'pred_class_name', 'prob']]
    results_path =os.path.split(os.path.normpath(args.preds_path))[0]
    fp_out = os.path.join(results_path, f'{args.output_name}.csv')
    df_updated.to_csv(fp_out, sep=',', header=True, index=False)

    print(f'Saved to {fp_out}', df_updated.head())

    return 0



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds_path', type=str, required=True,
                        help='path to ind_preds.csv (results_train/dataset_model/ind_preds.csv)')
    parser.add_argument('--save_wrong_preds_only', action='store_false',
                        help='by default only saves wrong preds (if use flag saves all)')
    parser.add_argument('--save_prob_th', type=float, default=80,
                        help='float from 0 to 1 (saves only preds with confidence above threshold)')
    parser.add_argument('--output_name', type=str, default='wrong_preds')

    parser.add_argument('--dataset_name', default=None, type=str, help='dataset name')
    parser.add_argument('--dataset_root_path', type=str, default=None,
                        help='the root directory for where the data/feature/label files are')
    # folders with images (can be same: those where it's all stored in 'data')
    parser.add_argument('--folder_val', type=str, default='data',
                        help='the directory where images are stored, ex: dataset_root_path/val/')
    parser.add_argument('--folder_test', type=str, default='data',
                        help='the directory where images are stored, ex: dataset_root_path/test/')
    # df files with img_dir, class_id
    parser.add_argument('--df_val', type=str, default='val.csv',
                        help='the df csv with img_dirs, targets, def: val.csv')
    parser.add_argument('--df_test', type=str, default='test.csv',
                        help='the df csv with img_dirs, targets, root/test.csv')
    parser.add_argument('--df_classid_classname', type=str, default='classid_classname.csv',
                        help='the df csv with classnames and class ids, root/classid_classname.csv')

    parser.add_argument('--train_trainval', action='store_false',
                        help='when true uses trainval for train and evaluates on test \
                        otherwise use train for train and evaluates on val')

    parser.add_argument("--cfg", type=str,
                        help="If using it overwrites args and reads yaml file in given path")

    args = parser.parse_args()

    if args.cfg:
        config = yaml_config_hook(os.path.abspath(args.cfg))
        for k, v in config.items():
            if hasattr(args, k):
                setattr(args, k, v)

    update_ind_preds(args)


main()
