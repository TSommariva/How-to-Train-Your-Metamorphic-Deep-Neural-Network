#!/bin/bash

#SBATCH --job-name=Nerf_WU10
#SBATCH --output=log/AaITERATIVE/CustomInit_4batchAccumulation_lr8.5e-4_warmup10_%j.out
#SBATCH  --error=log/AaITERATIVE/CustomInit_4batchAccumulation_lr8.5e-4_warmup10_%j.err
#SBATCH --time=24:00:00                         
#SBATCH --constraint="gpu_A40_48G|gpu_RTX6000_24G|gpu_RTXA5000_24G|gpu_RTX5000_16G"

#SBATCH --gres=gpu:1                            
#SBATCH --cpus-per-task=1

#SBATCH --account=tesi_tsommariva               
#SBATCH --partition=all_usr_prod                
#SBATCH --mail-type=END,FAIL                    
#SBATCH --mail-user=ts.slurm@gmail.com          

. /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate neumeta
# Command to execute Python program
export PYTHONPATH=/homes/tsommariva/neumeta:$PYTHONPATH
python3 /homes/tsommariva/neumeta/neumeta/train_cifar100_iterative.py --config /homes/tsommariva/neumeta/neumeta/config/cifar100/Cifar100_resnet56_myConf_Iterative.yaml \
    #--resume_from "/work/tesi_tsommariva/experiments/AaITERATIVE/SingleBlock_CustomInit_BatchAccumulationSteps:4_ArchAccumulationSteps:1_lr:0.00085/fineTuning/cifar100_nerf_best.pth" \