CUDA_VISIBLE_DEVICES=7  python attack.py -e attack_just_try -p  /data2/haoran/HW/HomeworkProject/experiments/just_try/ckpt/ckpt_epoch_63

CUDA_VISIBLE_DEVICES=7  python attack.py -e attack_just_try -p  /data2/haoran/HW/HomeworkProject/experiments/just_new/ckpt/ckpt_epoch_61

CUDA_VISIBLE_DEVICES=7  python attack.py -e attack_just_try\
 -p /data2/haoran/HW/HomeworkProject/experiments/pgd_train_just_try_step4/ckpt/ckpt_epoch_299


CUDA_VISIBLE_DEVICES=7  python attack.py -e attack_just_try\
 -p /data2/haoran/HW/HomeworkProject/experiments/pgd_train_quan_try/ckpt/ckpt_epoch_10

 CUDA_VISIBLE_DEVICES=4  python attack.py -e attack_just_try\
 -p /data2/haoran/HW/HomeworkProject/experiments/new_small_noattack_noquantization/ckpt/ckpt_epoch_100\
 -small_model

CUDA_VISIBLE_DEVICES=4  python attack.py -e attack_just_try\
 -p /data2/haoran/HW/HomeworkProject/experiments/new_small_attack_noquantization/ckpt/ckpt_epoch_100\
 -small_model

 CUDA_VISIBLE_DEVICES=4  python attack.py -e attack_just_try\
 -p /data2/haoran/HW/HomeworkProject/experiments/new_quant_noattack_quantization/ckpt/ckpt_epoch_100\
 -small_model

  CUDA_VISIBLE_DEVICES=4  python attack.py -e attack_just_try\
 -p /data2/haoran/HW/HomeworkProject/experiments/new_quant_attack_quantization/ckpt/ckpt_epoch_100\
 -quant_model

  CUDA_VISIBLE_DEVICES=4  python attack.py -e attack_just_try\
 -p /data2/haoran/HW/HomeworkProject/experiments/new_small_orthogonal/ckpt/ckpt_epoch_100\
  -small_model

   CUDA_VISIBLE_DEVICES=4  python attack.py -e attack_just_try\
 -p /data2/haoran/HW/HomeworkProject/experiments/new_RSE/ckpt/ckpt_epoch_100 -RSE

 
   CUDA_VISIBLE_DEVICES=7  python attack.py -e attack_just_try\
 -p /data2/haoran/HW/HomeworkProject/experiments/new_small_spectral_new/ckpt/ckpt_epoch_100 -small_model