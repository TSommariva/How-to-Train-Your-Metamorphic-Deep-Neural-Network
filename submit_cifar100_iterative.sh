#!/bin/bash

#SBATCH --job-name=NeRF
#SBATCH --output=log/ResnetXL_NoEMA/NewExt_DimStep4_ALLSimultaneusBlocks_WarmupAnnealing_%j.out
#SBATCH  --error=log/ResnetXL_NoEMA/NewExt_DimStep4_ALLSimultaneusBlocks_WarmupAnnealing_%j.err
#SBATCH --time=1:00:00                         
#SBATCH --constraint="gpu_L40S_48G|gpu_A40_48G" #|gpu_RTXA5000_24G|gpu_RTX6000_24G" #|gpu_RTX5000_16G" #|gpu_2080Ti_11G"
#SBATCH --mem=48G

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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/homes/tsommariva/neumeta:$PYTHONPATH

python3 /homes/tsommariva/neumeta/neumeta/train_cifar100_iterative.py --config neumeta/config/cifar100/Cifar100_resnet56_myConf_Iterative.yaml \
    --resume_from "/work/tesi_tsommariva/experiments/AaITERATIVE/RXL_OnlyLast:False_8simultaneusBlocks_DimStep4_50_warmup_cosine_20e_lr0.00085_etaMin0.0001/block7/cifar100_nerf_last.pth"
    #--model.only_last=False \
    #--experiment.name="Debug_NoValidateTrain" \
    #--model.only_last=False \
    #--training.scheduler="warmup_const_cosine" \
