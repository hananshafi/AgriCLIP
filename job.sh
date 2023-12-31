#!/bin/bash
#SBATCH --job-name=fish10
#SBATCH -N1
#SBATCH -n1
#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --error=./out_logs/fish10.err
#SBATCH --output=./out_logs/fish10.out

ulimit -S -u 5000
cd /l/users/hanan.ghani/mae
torchrun main_finetune.py --model_type dinov2 --batch_size 64 --data_path /l/users/muzammal.naseer/umair-data/down-stream/Fish/Tasks/Classification/Dataset-10/DeepFish/Classification --epochs 100 --blr 5e-4 --layer_decay 0.65 --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 --nb_classes 2 --output_dir dataset10 --dataset_type fish10
