import os
import glob

import pandas as pd
from PIL import Image
import torch

from fgir_kd.other_utils.build_args import parse_inference_args
from fgir_kd.data_utils.build_transform import build_transform

from vis_dfsm import build_environment_inference


def prepare_img(fn, args, transform):
    # open img
    img = Image.open(fn).convert('RGB')
    # Preprocess image
    img = transform(img).unsqueeze(0).to(args.device)
    return img


def search_images(folder):
    # the tuple of file types
    types = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')

    # if folder is a file
    if os.path.isfile(folder):
        # if folder is a .txt or .csv with file names
        if os.path.splitext(folder)[1] in ('.txt', '.csv'):
            df = pd.read_csv(folder)
            print('Total image files', len(df))
            return df['dir'].tolist()

        # if folder is a path to an image
        elif any([t.replace('*', '') in os.path.splitext(folder)[1] for t in types]):
            return [folder]

    # else if directory
    files_all = []
    for file_type in types:
        # files_all is the list of files
        path = os.path.join(folder, '**', file_type)
        files_curr_type = glob.glob(path, recursive=True)
        files_all.extend(files_curr_type)

        print(file_type, len(files_curr_type))

    print('Total image files', len(files_all))
    return files_all


def prepare_inference(args):
    _, _, hook, amp_autocast = build_environment_inference(args)

    transform = build_transform(args=args, split='test')

    # Load class names
    dic_classid_classname = None

    if args.dataset_root_path and args.df_classid_classname:
        fp = os.path.join(args.dataset_root_path, args.df_classid_classname)

        if os.path.isfile(fp):
            dic_classid_classname = pd.read_csv(fp, index_col='class_id')['class_name'].to_dict()
    
    return hook, amp_autocast, transform, dic_classid_classname


def inference_single(args, amp_autocast, hook, img, dic_classid_classname=None,
                     file=None, save=False):

    with amp_autocast():
        if save:
            pow = '_power' if args.vis_mask_pow else ''
            fn = os.path.splitext(os.path.split(file)[1])[0]
            fp = os.path.join(args.results_dir, f'{fn}_{args.vis_mask}{pow}.png')

            preds, pil_image = hook.inference_save_vis(
                img, fp, args.vis_pred_text, args.custom_mean_std,
                args.vis_mask_pow, 1, args.vis_reg_reduction)

        else:
            preds, _ = hook.inference(img)

    preds = preds.squeeze(0)
    for i, idx in enumerate(torch.topk(preds, k=args.top_k).indices.tolist()):
        prob = torch.softmax(preds, -1)[idx].item()
        if dic_classid_classname is not None:
            classname = dic_classid_classname[idx]
            out_text = '[{idx}] {label:<75} ({p:.2f}%)'.format(idx=idx, label=classname, p=prob*100)
            print(out_text)
        else:
            out_text = '[{idx}] ({p:.2f}%)'.format(idx=idx, p=prob*100)
            print(out_text)
        if i == 0:
            top1_text = out_text

    if save:
        return top1_text, pil_image
    return top1_text


def inference_all(args):
    files_all = search_images(args.images_path)

    hook, amp_autocast, transform, dic_classid_classname = prepare_inference(args)

    for file in files_all:
        print(file)
        img = prepare_img(file, args, transform)

        # Classify
        inference_single(args, amp_autocast, hook, img, dic_classid_classname,
                         file, save=args.vis_mask)

    return 0


def main():
    args = parse_inference_args()
    args.try_fused_attn = False
    args.debugging = True

    inference_all(args)

    return 0


if __name__ == '__main__':
    main()
