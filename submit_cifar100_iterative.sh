#!/bin/bash

#SBATCH --job-name=Nerf
#SBATCH --output=log/SkipInit/DictNerf_SkipInit_AeB_%j.out
#SBATCH  --error=log/SkipInit/DictNerf_SkipInit_AeB_%j.err
#SBATCH --time=24:00:00                         
#SBATCH --constraint="gpu_A40_48G|gpu_RTXA5000_24G|gpu_RTX6000_24G" #|gpu_RTX5000_16G|gpu_2080Ti_11G"

#SBATCH --gres=gpu:1                            
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6

#SBATCH --account=tesi_tsommariva               
#SBATCH --partition=all_usr_prod                
#SBATCH --mail-type=ALL                   
#SBATCH --mail-user=ts.slurm@gmail.com          

. /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate neumeta
# Command to execute Python program

LR=4e-5

export PYTHONPATH=/homes/tsommariva/neumeta:$PYTHONPATH
python3 /homes/tsommariva/neumeta/neumeta/train_cifar100_iterative.py --config /homes/tsommariva/neumeta/neumeta/config/cifar100/Cifar100_resnet56_myConf_Iterative.yaml \
   --resume_from "/work/tesi_tsommariva/experiments/AaITERATIVE/skipInitAeB_clsLr0.001_resmlpDict_Scalar0_4kernelGroups_4AccumulationSteps_50e_lr0.00085_bs128_warmup_cosine20/block3/cifar100_nerf_best.pth" \
#    --training.learning_rate=$LR \