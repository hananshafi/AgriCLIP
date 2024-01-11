#!/bin/bash
#SBATCH --job-name=clip
#SBATCH -N1
#SBATCH -n1
#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --error=./out_logs/clip.err
#SBATCH --output=./out_logs/clip.out

ulimit -S -u 5000
cd /l/users/hanan.ghani/mae
torchrun finetune_clip.py  --batch_size 8 --data_path /l/users/muzammal.naseer/umair-data/down-stream/Fish/Tasks/Classification/Dataset-4/NA_Fish_Dataset --epochs 100 --blr 5e-4 --layer_decay 0.65 --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 --nb_classes 9 --output_dir ./finetuning_results/clip --dataset_type clip
