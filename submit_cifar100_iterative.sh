#!/bin/bash

#SBATCH --job-name=Nerf
#SBATCH --output=log/AaITERATIVE/Nerf_diffShapes_nocustomInit_%j.out
#SBATCH  --error=log/AaITERATIVE/Nerf_diffShapes_nocustomInit_%j.err
#SBATCH --time=11:00:00                         
#SBATCH --constraint="gpu_A40_48G|gpu_RTX6000_24G|gpu_RTXA5000_24G|gpu_RTX5000_16G|gpu_2080Ti_11G"

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
lr=5e-4
eta_min=5e-5
export PYTHONPATH=/homes/tsommariva/neumeta:$PYTHONPATH
python3 /homes/tsommariva/neumeta/neumeta/train_cifar100_iterative.py --config /homes/tsommariva/neumeta/neumeta/config/cifar100/Cifar100_resnet56_myConf_Iterative.yaml \
    #--training.learning_rate=$lr \
    #--training.eta_min=$eta_min \
    #--training.scheduler='cosine' \
    #--resume_from "/work/tesi_tsommariva/experiments/AaITERATIVE/FullTrainingFrom1_1AccumulationSteps_CustomInit:True_singleBlock:False_500e_lr0.0003_ftScaling:False:0.1/block5/cifar100_nerf_best.pth" \