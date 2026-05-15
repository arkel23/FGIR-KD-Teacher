import os
import numpy as np
import gradio as gr

from fgir_kd.other_utils.build_args import parse_inference_args, yaml_config_hook
from inference import prepare_inference, prepare_img, inference_single


def adjust_demo_args(args, dataset, model, vis_mask):
    args.try_fused_attn = False
    args.debugging = True

    args.image_size = 448
    args.test_resize_size = int(args.image_size * 1.34)
    args.test_square_resize_center_crop = True
    args.convert_cal_student_keep_head = True

    fn = f'{dataset}_{model}_resnet101_cal_tgda_101.pth'
    args.ckpt_path = os.path.join('..', '..', 'results_vitfs', 'serial101', fn)

    args.cfg = os.path.join('configs', f'{dataset}_weakaugs.yaml')

    if args.cfg:
        config = yaml_config_hook(os.path.abspath(args.cfg))
        for k, v in config.items():
            if hasattr(args, k):
                setattr(args, k, v)

    args.dataset_root_path = os.path.join('data', dataset)
    args.model_name = model
    args.vis_mask = vis_mask
    return args


def demo(
    file_path,
    dataset='cub', 
    model='vitfs_tiny_patch16_gap_reg4_dinov2_bn_init',
    vis_mask='rollout',
    ):
    args = parse_inference_args()

    args = adjust_demo_args(args, dataset, model, vis_mask)

    # prepare model and transform for inference
    hook, amp_autocast, transform, dic_classid_classname = prepare_inference(args)

    # prepare each image for inference: transform and make into batch of 1
    img = prepare_img(file_path, args, transform)

    # Classify
    top1_text, masked_image =  inference_single(
        args, amp_autocast, hook, img, dic_classid_classname, 'temp.png', vis_mask)

    masked_image = np.array(masked_image)

    return top1_text, masked_image


title = 'Fine-Grained Image Recognition'
description = 'Demo for "Fine-Grained Image Recognition"'
article = '''<p style='text-align: center'>
    Fine-Grained Image Recognition 
    </p>'''

inputs = [
    gr.components.Image(
        type='filepath', label='Input image'
    ),
    gr.components.Radio(
        value='cub',
        choices=['aircraft', 'cars', 'cub'],
        label='Dataset (def: cub)'
    ),
    gr.components.Radio(
        value='vitfs_tiny_patch16_gap_reg4_dinov2_bn_init',
        choices=['vitfs_tiny_patch16_gap_reg4_dinov2_bn_init'],
        label='Model name'
    ),
    gr.components.Radio(
        value='rollout',
        choices=['rollout', 'attention_0', 'attention_11'],
        # choices=['CAM', 'GradCAM', 'rollout', 'attention_0'],
        label='Decision interpretation method'
    ),
]

outputs = [
    gr.components.Textbox(label='Predicted class and tags'),
    gr.components.Image(label='Decision Heatmap')
]

examples = [
    [os.path.join('samples', 'others', 'cub_black_footed_albatross.jpg')],
    [os.path.join('samples', 'others', 'cub_common_yellowthroat.jpg')],
]

gr.Interface(
    demo, inputs, outputs, title=title, description=description,
    article=article, examples=examples).launch(debug=True, share=True)
