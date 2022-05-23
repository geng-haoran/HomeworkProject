CUDA_VISIBLE_DEVICES=6 python train.py -e pgd_train_just_try_step4 -t 300 \
-a -c

CUDA_VISIBLE_DEVICES=6 python train.py -e pgd_train_noattack -t 300 \
 -c

CUDA_VISIBLE_DEVICES=5 python train.py -e pgd_train_just_try_step4_3-255 -t 300 \
-a -sm

CUDA_VISIBLE_DEVICES=5 python train.py -e pgd_train_quan_try -t 300 \
 -quant_model -attack -quantization -continue


 CUDA_VISIBLE_DEVICES=4 python train.py -e new_small_noattack_noquantization -t 300 \
 -small_model

 CUDA_VISIBLE_DEVICES=4 python train.py -e new_small_attack_noquantization -t 300 \
 -small_model -attack


 CUDA_VISIBLE_DEVICES=7 python train.py -e new_quant_noattack_quantization -t 300 \
 -quant_model -quantization

  CUDA_VISIBLE_DEVICES=4 python train.py -e new_quant_attack_quantization_relu -t 300 \
 -quant_model -quantization -attack

 
 CUDA_VISIBLE_DEVICES=6 python train.py -e new_small_orthogonal_new -t 300 \
 -small_model -orthogonal

  CUDA_VISIBLE_DEVICES=6 python train.py -e new_small_spectral_new -t 300 \
 -small_model -spectral

  CUDA_VISIBLE_DEVICES=4 python train.py -e new_RSE -t 300 \
 -RSE