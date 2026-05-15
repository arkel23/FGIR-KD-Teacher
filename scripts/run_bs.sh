#!/bin/bash

run='False'
run_lr='False'
run_seed='False'
augs='weakaugs'
ls=''
sd=''
freeze_backbone=''

device=0
batch_size=64
serial=1
seed=1
lr=0.003

dataset_name='cub'
model_name='vit_b16 --classifier cls --cfg_method configs/methods/glsim.yaml'

lr_array=('0.03' '0.01' '0.003' '0.001')
seed_array=('1' '10')

VALID_ARGS=$(getopt  -o '' --long run,run_lr,run_seed,med_augs,ls,sd,freeze_backbone,device:,batch_size:,serial:,seed:,lr:,dataset_name:,model_name: -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1;
fi

eval set -- "$VALID_ARGS"
while [ : ]; do
  case "$1" in
    --run)
        run='True'
        shift 1
        ;;
    --run_lr)
        run_lr='True'
        shift 1
        ;;
    --run_seed)
        run_seed='True'
        shift 1
        ;;
    --med_augs)
        augs='medaugs'
        shift 1
        ;;
    --ls)
        ls=' --ls'
        shift 1
        ;;
    --sd)
        sd=' --sd 0.1'
        shift 1
        ;;
    --freeze_backbone)
        freeze_backbone=' --freeze_backbone'
        lr_array=('0.3' '0.1' '0.03' '0.01' '0.003')
        shift 1
        ;;
    --device)
        device=${2}
        shift 2
        ;;
    --batch_size)
        batch_size=${2}
        shift 2
        ;;
    --serial)
        serial=${2}
        shift 2
        ;;
    --seed)
        seed=${2}
        shift 2
        ;;
    --lr)
        lr=${2}
        shift 2
        ;;
    --dataset_name)
        dataset_name=${2}
        shift 2
        ;;
    --model_name)
        model_name=${2}
        shift 2
        ;;
    --) shift;
        break
        ;;
  esac
done

# CMD="CUDA_VISIBLE_DEVICES=${device} nohup python -u tools/train.py --serial ${serial} --cfg configs/${dataset_name}_ft_medaugs.yaml${is_448}"
CMD="nohup python -u tools/train.py --serial ${serial} --batch_size ${batch_size} --cfg configs/${dataset_name}_ft_${augs}.yaml${ls}${sd}${freeze_backbone}"
echo "${CMD}"

# single run
if [[ "$run" == "True" ]]; then
    echo "${CMD} --seed ${seed} --base_lr ${lr} --model_name ${model_name}"
    ${CMD} --seed ${seed} --base_lr ${lr} --model_name ${model_name}
fi

# lr run
if [[ "$run_lr" == "True" ]]; then
    for rate in ${lr_array[@]}; do
        echo "${CMD} --seed ${seed} --base_lr ${rate} --model_name ${model_name} --train_trainval"
        ${CMD} --seed ${seed} --base_lr ${rate} --model_name ${model_name} --train_trainval
    done
fi


# seed run
if [[ "$run_seed" == "True" ]]; then
    for seed in ${seed_array[@]}; do
        echo "${CMD} --seed ${seed} --base_lr ${lr} --model_name ${model_name}"
        ${CMD} --seed ${seed} --base_lr ${lr} --model_name ${model_name}
    done
fi
