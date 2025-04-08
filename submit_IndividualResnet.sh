#!/bin/bash

#SBATCH --job-name=Individualbs
#SBATCH --output=log/AaPaper/Ablation/Individual/priorLarge_sgd_%j.out
#SBATCH  --error=log/AaPaper/Ablation/Individual/priorLarge_sgd_%j.err
#SBATCH --time=8:00:00                         
#SBATCH --constraint="gpu_L40S_48G|gpu_A40_48G|gpu_RTXA5000_24G|gpu_RTX6000_24G|gpu_RTX5000_16G|gpu_2080Ti_11G"
##SBATCH --mem=20G

#SBATCH --gres=gpu:1                            
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

#SBATCH --account=tesi_tsommariva               
#SBATCH --partition=all_usr_prod                
#SBATCH --mail-type=ALL                   
#SBATCH --mail-user=ts.slurm@gmail.com          

. /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate neumeta
# Command to execute Python program

#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/homes/tsommariva/slim_neumeta/tesi:$PYTHONPATH

python3  /homes/tsommariva/neumeta/neumeta/trainResnet56.py --config /homes/tsommariva/neumeta/neumeta/config/cifar100/Cifar100_resnet56.yaml