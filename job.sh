#!/bin/bash
#SBATCH --job-name=fish4
#SBATCH -N1
#SBATCH -n1
#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --error=./out_logs/fish4.err
#SBATCH --output=./out_logs/fish4.out

cd /l/users/hanan.ghani/mae
torchrun main_finetune.py --model_type mae --model vit_base_patch16 --batch_size 64 --finetune ./checkpoints/mae_pretrain_vit_base.pth --data_path /l/users/muzammal.naseer/umair-data/down-stream/Fish/Tasks/Classification/Dataset-4/NA_Fish_Dataset --epochs 100 --blr 5e-4 --layer_decay 0.65 --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 --nb_classes 9 --output_dir /l/users/hanan.ghani/mae/70_30_finetuning/dataset4 --dataset_type fish4
