# train
CUDA_VISIBLE_DEVICES=0 \
python cifar10_train.py \
  --data_dir=/tmp/my_cifar10_data_single_gpu \
  --train_dir=/tmp/my_single_gpu_train \
  --max_steps=1000 

CUDA_VISIBLE_DEVICES=0 \
python cifar10_eval.py \
  --data_dir=/tmp/my_cifar10_data_single_gpu \
  --checkpoint_dir=/tmp/my_single_gpu_train \
  --eval_dir=/tmp/my_single_gpu_eval

