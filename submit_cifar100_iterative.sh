#!/bin/bash

#SBATCH --job-name=NeRF
#SBATCH --output=log/AaPaper/Backbone_150e_%j.out
#SBATCH  --error=log/AaPaper/Backbone_150e_%j.err
#SBATCH --time=18:20:00                         
#SBATCH --constraint="gpu_L40S_48G|gpu_A40_48G" #|gpu_RTXA5000_24G|gpu_RTX6000_24G" #|gpu_RTX5000_16G" #|gpu_2080Ti_11G"
##SBATCH --mem=60G

#SBATCH --gres=gpu:1                            
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

#SBATCH --account=tesi_tsommariva               
#SBATCH --partition=all_usr_prod                
#SBATCH --mail-type=ALL                   
#SBATCH --mail-user=ts.slurm@gmail.com          

. /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate neumeta
# Command to execute Python program

#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/homes/tsommariva/neumeta:$PYTHONPATH

python3 /homes/tsommariva/neumeta/neumeta/train_cifar100_iterative.py --config neumeta/config/cifar100/Cifar100_resnet56_myConf_Iterative.yaml \
    --resume_from "/work/tesi_tsommariva/experiments/AaITERATIVE/resmlpDict_CommonB8_8l_512hd_32f_8simultaneusBlocks_50_4AccumulationSteps_warmup_cosine_20e/nerf_block1.pth" \
    #--experiment.num_epochs=85 \
    #--training.scheduler="warmup_const_cosine" \