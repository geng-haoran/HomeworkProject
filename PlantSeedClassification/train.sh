CUDA_VISIBLE_DEVICES=7 \
python train.py -e train_1e-3_useNorm_Adam -c
 

CUDA_VISIBLE_DEVICES=6 \
python train.py -e train_1e-3_noNorm_Adam -c


CUDA_VISIBLE_DEVICES=4 \
python train.py -e train_1e-3_noNorm_SGD  -o SGD

CUDA_VISIBLE_DEVICES=4 \
python train.py -e train_1e-3_noNorm_Adagrad  -o Adagrad



CUDA_VISIBLE_DEVICES=5 python train.py -e train_1e-3_VGGNoAug_Adam  -o Adam -m vgg16