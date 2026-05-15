# Teacher Guided Data Augmentation

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

To train a `ViT FS` with `TGDA` on CUB using image size 448 for teacher and image size 224 for student:

```
python -u tools/train_student.py --selector cal --tgda  --image_size 448 --serial 201 --sd 0.1 --ls --trivial_aug --model_name_teacher resnet101 --model_name vitfs_tiny_patch16_gap_reg4_dinov2_bn_init --cfg configs/cub_weakaugs.yaml --ckpt_path_teacher /edahome/pcslab/pcs05/edwin/results_backbones/serial15_ckpts/cub_tv_resnet101_cal_is448.pth --student_image_size 224
```

## Compute CKA Similarity

For frozen vanilla ViT B-16
```
python -u tools/postprocess/compute_cka_dists_attn_mean_std.py --cfg configs/cotton_weakaugs.yaml --debugging --batch_size 8 --model_name vit_b16 --pretrained --fp16
```

```
python -u tools/postprocess/compute_cka_dists_two_models.py --cfg configs/cotton_weakaugs.yaml --debugging --batch_size 8 --model_name vit_b16 --pretrained --fp16 --model_name_teacher resnet50.tv_in1k
```

Results will be saved in `results_inference` directory.


## Visualize attention

For ViT with ILA to visualize attention rollout for the 1st encoder group (first 4 encoder blocks: 0_4):

```
python -u tools/postprocess/vis_dfsm.py --cfg configs/cub_weakaugs.yaml --debugging --batch_size 12 --vis_cols 12 --vis_mask attention_0 --model_name vit_b16 --pretrained
```


## Evaluation

To evaluate a particular checkpoint on the test set (logs results to W&B):

```
python tools/train.py --ckpt_path ckpts/cub_glsim_224.pth --test_only
```

To visulize misclassification for a particular network on the test set:

```
python -u tools/postprocess/vis_dfsm.py --cfg configs/cub_weakaugs.yaml --debugging --batch_size 12 --vis_cols 12 --vis_mask attention_0 --model_name vit_b16 --pretrained --vis_wrong_only
```

For a specific class can use `--vis_class {CLASS_ID}`



# Citation
If you find our work helpful in your research, please cite it as:
```
Pending
```

# Acknowledgements
We thank NYCU's HPC Center and National Center for High-performance Computing (NCHC) for providing computational and storage resources. 

We thank the authors of [TransFG](https://github.com/TACJu/TransFG), [FFVT](https://github.com/Markin-Wang/FFVT), [SimTrans](https://github.com/PKU-ICST-MIPL/SIM-Trans_ACMMM2022), [CAL](https://github.com/raoyongming/CAL), [MPN-COV](https://github.com/jiangtaoxie/MPN-COV), [VPT](https://github.com/KMnP/vpt), [VQT](https://github.com/andytu28/VQT), [ConvPass](https://github.com/JieShibo/PETL-ViT/tree/main/convpass) and [timm](https://github.com/huggingface/pytorch-image-models/) for providing implementations for comparison.

Also, [Weight and Biases](https://wandb.ai/) for their platform for experiment management.
 
