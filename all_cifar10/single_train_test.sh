# train
CUDA_VISIBLE_DEVICES=0 \
python cifar10_train.py \
  --data_dir=/tmp/cifar10_data \
  --train_dir=/tmp/all_single_gpu_train \
  --max_steps=1000 

CUDA_VISIBLE_DEVICES=0 \
python cifar10_eval.py \
  --data_dir=/tmp/cifar10_data \
  --checkpoint_dir=/tmp/all_single_gpu_train \
  --eval_dir=/tmp/all_single_gpu_eval

