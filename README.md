# How to Choose Your Teacher for Fine Grained Image Recognition

Official PyTorch code for the paper: [How to Choose Your Teacher for Fine Grained Image Recognition published in Fine-Grained Visual Categorization](https://arxiv.org/abs/2605.15689) (FGVC13) @ CVPR 2026 Workshop.

This paper introduces a teacher selection metric, Ratio 1-2, based on teacher prediction ratios.
![](assets/metric_new.png)

Our proposed metric demonstrates a strong correlation with the resulting student performance during knowledge distillation.
![](assets/202_ratio_1_2_normal_cub.png)

Extensive experiments across eight fine-grained image recognition (FGIR) datasets show that our method consistently achieves favorable results.
![](assets/results.png)

## Setup
```
pip install -e . 
```

## Preparation

Datasets are downloaded from:
```
Xiaohan Yu, Yang Zhao, Yongsheng Gao, Xiaohui Yuan, Shengwu Xiong (2021). Benchmark Platform for Ultra-Fine-Grained Visual Categorization BeyondHuman Performance. In ICCV2021.
https://github.com/XiaohanYu-GU/Ultra-FGVC?tab=readme-ov-file
```

Visualize with:

```
python -u tools/postprocess/vis_dfsm.py --cfg configs/cub_weakaugs.yaml --debugging --batch_size 12 --vis_cols 12
```

To visualize a specific class add: ` --vis_class {CLASS_ID}`.

## Train

To train a `levit_128s` on aircraft using image size 224 for teacher and image size 224 for student:

```
python -u tools/train_student.py --project_name KD_ST_Others --square_resize_random_crop --test_square_resize_center_crop --cpu_workers 28 --cfg configs/aircraft_weakaugs.yaml --model_name levit_128s --model_name_teacher vit_b16 --ckpt_path_teacher ../../results_backbones/fz_ckpts/aircraft_vit_b16_fz.pth --epochs 1 --opt adamw --weight_decay 5e-2 --lr 5e-3 --temp 2 --loss_kd_weight 10 --compute_train_wise_teacher_metrics --serial 999
```

more examples of how to run the code may can be seen in scripts/sample.sh


# Citation
If you find our work helpful in your research, please cite it as:
```
Pending
```

# Acknowledgements
We thank NYCU's HPC Center and National Center for High-performance Computing (NCHC) for providing computational and storage resources. 

We thank the authors of [TransFG](https://github.com/TACJu/TransFG), [FFVT](https://github.com/Markin-Wang/FFVT), [SimTrans](https://github.com/PKU-ICST-MIPL/SIM-Trans_ACMMM2022), [CAL](https://github.com/raoyongming/CAL), [MPN-COV](https://github.com/jiangtaoxie/MPN-COV), [VPT](https://github.com/KMnP/vpt), [VQT](https://github.com/andytu28/VQT), [ConvPass](https://github.com/JieShibo/PETL-ViT/tree/main/convpass) and [timm](https://github.com/huggingface/pytorch-image-models/) for providing implementations for comparison.

Also, [Weight and Biases](https://wandb.ai/) for their platform for experiment management.
 
