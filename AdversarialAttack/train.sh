CUDA_VISIBLE_DEVICES=6 python train.py -e pgd_train_just_try_step4 -t 300 \
-a -c

CUDA_VISIBLE_DEVICES=6 python train.py -e pgd_train_noattack -t 300 \
 -c

CUDA_VISIBLE_DEVICES=5 python train.py -e pgd_train_just_try_step4_3-255 -t 300 \
-a -sm

CUDA_VISIBLE_DEVICES=5 python train.py -e pgd_train_quan_try -t 300 \
 -quant_model -attack -quantization -continue