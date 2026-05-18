# Train FZ
python -u tools/train_student.py --project_name KD_ST_Others --square_resize_random_crop --test_square_resize_center_crop --cpu_workers 28 --cfg configs/aircraft_weakaugs.yaml --model_name levit_128s --model_name_teacher vit_b16 --ckpt_path_teacher ../../results_backbones/fz_ckpts/aircraft_vit_b16_fz.pth --epochs 1 --opt adamw --weight_decay 5e-2 --lr 5e-3 --temp 2 --loss_kd_weight 10 --compute_train_wise_teacher_metrics --serial 999

# Train FT
python -u tools/train_student.py --project_name KD_ST_Others --square_resize_random_crop --test_square_resize_center_crop --cpu_workers 28 --cfg configs/aircraft_weakaugs.yaml --model_name levit_128s --model_name_teacher vit_b16 --ckpt_path_teacher ../../results_backbones/ft_ckpts/aircraft_vit_b16.pth --epochs 1 --opt adamw --weight_decay 5e-2 --lr 5e-3 --temp 2 --loss_kd_weight 10 --compute_train_wise_teacher_metrics --serial 999

# Train CAL
python -u tools/train_student.py --project_name KD_ST_Others --square_resize_random_crop --test_square_resize_center_crop --cpu_workers 28 --cfg configs/aircraft_weakaugs.yaml --model_name levit_128s --model_name_teacher vit_b16 --ckpt_path_teacher ../../results_backbones/cal_ckpts/aircraft_vit_b16_cal.pth --selector cal --epochs 1 --opt adamw --weight_decay 5e-2 --lr 5e-3 --image_size 224 --temp 2 --loss_kd_weight 10 --compute_train_wise_teacher_metrics --serial 999