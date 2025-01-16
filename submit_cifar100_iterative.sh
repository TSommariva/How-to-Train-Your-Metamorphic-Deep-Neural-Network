#!/bin/bash

#SBATCH --job-name=Dict
#SBATCH --output=log/DimDict/DictNerf_KernelGroups_%j.out
#SBATCH  --error=log/DimDict/DictNerf_KernelGroups_%j.err
#SBATCH --time=6:00:00                         
#SBATCH --constraint="gpu_A40_48G|gpu_RTX5000_16G|gpu_RTXA5000_24G|gpu_RTX6000_24G" #|gpu_2080Ti_11G"

#SBATCH --gres=gpu:1                            
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

#SBATCH --account=tesi_tsommariva               
#SBATCH --partition=all_usr_prod                
#SBATCH --mail-type=BEGIN,END,FAIL                    
#SBATCH --mail-user=ts.slurm@gmail.com          

. /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate neumeta
# Command to execute Python program

LR=4e-5

export PYTHONPATH=/homes/tsommariva/neumeta:$PYTHONPATH
python3 /homes/tsommariva/neumeta/neumeta/train_cifar100_iterative.py --config /homes/tsommariva/neumeta/neumeta/config/cifar100/Cifar100_resnet56_myConf_Iterative.yaml \
    --training.learning_rate=$LR \
#   --resume_from "/work/tesi_tsommariva/experiments/AaITERATIVE/resmlpDict_4AccumulationSteps_50e_lr0.00085_bs128_warmup_cosine_0.00085/block8/cifar100_nerf_best.pth" \