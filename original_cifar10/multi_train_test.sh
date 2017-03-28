# train
CUDA_VISIBLE_DEVICES=0 \
python cifar10_multi_gpu_train.py \
  --data_dir=/tmp/multi_gpu_cifar10_data \
  --train_dir=/tmp/multi_gpu_train \
  --max_steps=1000 \
  --num_gpus=1

CUDA_VISIBLE_DEVICES=0 \
python cifar10_eval.py \
  --data_dir=/tmp/multi_gpu_cifar10_data \
  --checkpoint_dir=/tmp/multi_gpu_train \
  --eval_dir=/tmp/multi_gpu_eval

