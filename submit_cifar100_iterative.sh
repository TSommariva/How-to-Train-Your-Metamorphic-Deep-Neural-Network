#!/bin/bash

#SBATCH --job-name=DecLT
#SBATCH --output=log/HybridTraining/150e_dec_%j.out
#SBATCH  --error=log/HybridTraining/150e_dec_%j.err
#SBATCH --time=24:00:00                         
#SBATCH --constraint="gpu_A40_48G|gpu_RTXA5000_24G|gpu_RTX6000_24G" #|gpu_RTX5000_16G" #|gpu_2080Ti_11G"

#SBATCH --gres=gpu:1                            
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

#SBATCH --account=tesi_tsommariva               
#SBATCH --partition=all_usr_prod                
#SBATCH --mail-type=ALL                   
#SBATCH --mail-user=ts.slurm@gmail.com          

. /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate neumeta
# Command to execute Python program
export PYTHONPATH=/homes/tsommariva/neumeta:$PYTHONPATH

python3 /homes/tsommariva/neumeta/neumeta/train_cifar100_iterative.py --config /homes/tsommariva/neumeta/neumeta/config/cifar100/Cifar100_resnet56_myConf_Iterative.yaml \
    --training.scheduler="warmup_const_cosine" \
#    --resume_from "/work/tesi_tsommariva/experiments/AaITERATIVE/HybridTraining_NoBackbone_resmlpDict_150_4AccumulationSteps_warmup_cosine_20e/cifar100_nerf_best.pth" \