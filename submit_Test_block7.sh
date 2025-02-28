#!/bin/bash

#SBATCH --job-name=NeRF_ResNetXL
#SBATCH --output=log/EVAL/DimStep4_AllBlocks_%j.out
#SBATCH  --error=log/EVAL/DimStep4_AllBlocks_%j.err
#SBATCH --time=10:00:00                         
#SBATCH --constraint="gpu_L40S_48G|gpu_A40_48G|gpu_RTXA5000_24G|gpu_RTX6000_24G" #|gpu_RTX5000_16G" #|gpu_2080Ti_11G"
#SBATCH --mem=32G

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

python3 /homes/tsommariva/neumeta/neumeta/testBloks.py --config neumeta/config/cifar100/Cifar100_resnet56_myConf_Iterative.yaml \
    --resume_from "/work/tesi_tsommariva/experiments/AaITERATIVE/NoEMA_OnlyLast_DimStep4_resmlpDictXL_50_4AccumulationSteps_warmup_cosine_20e" \
    --experiment.name="TEST_DimsMod4" \
    #--model.only_last=False \
    #--training.scheduler="warmup_const_cosine" \