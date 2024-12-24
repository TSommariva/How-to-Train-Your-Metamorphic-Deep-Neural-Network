#!/bin/bash

#SBATCH --job-name=Baseline          
#SBATCH --output=log/AaITERATIVE/4batchAccumulation_100eTmax100_4Blocks_iterative_BASELINEtest.out
#SBATCH --error=log/AaITERATIVE/4batchAccumulation_100eTmax100_4Blocks_iterative_BASELINEtest.err 
#SBATCH --time=12:00:00                         
#SBATCH --constraint="gpu_A40_48G|gpu_RTX6000_24G|gpu_RTXA5000_24G|gpu_RTX5000_16G"

#SBATCH --gres=gpu:1                            
#SBATCH --cpus-per-task=2

#SBATCH --account=tesi_tsommariva               
#SBATCH --partition=all_usr_prod                
#SBATCH --mail-type=END,FAIL                    
#SBATCH --mail-user=ts.slurm@gmail.com          

. /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate neumeta
# Command to execute Python program
export PYTHONPATH=/homes/tsommariva/neumeta:$PYTHONPATH
python3 /homes/tsommariva/neumeta/neumeta/train_cifar100.py --config /homes/tsommariva/neumeta/neumeta/config/cifar100/Cifar100_resnet56_myConf_Iterative.yaml \
    #--resume_from "experiments/AaITERATIVE/8Blocks_300e_4AccumulationSteps_bottomUp:True_Iterative:True_FINISHfromOldBlock2_150e/block2/cifar100_nerf_best.pth" \