#!/bin/bash

#SBATCH --job-name=SingleBlock           
#SBATCH --output=log/AaITERATIVE/SingleBlock_Finish_%A_%a.out
#SBATCH  --error=log/AaITERATIVE/SingleBlock_Finish_%A_%a.err
#SBATCH --time=24:00:00                         
#SBATCH --constraint="gpu_A40_48G|gpu_RTX6000_24G|gpu_RTXA5000_24G|gpu_RTX5000_16G"
#SBATCH --array=0-1
#SBATCH --gres=gpu:1                            
#SBATCH --cpus-per-task=1

#SBATCH --account=tesi_tsommariva               
#SBATCH --partition=all_usr_prod                
#SBATCH --mail-type=END,FAIL                    
#SBATCH --mail-user=ts.slurm@gmail.com          

. /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate neumeta

# Define accumulation steps configurations
case $SLURM_ARRAY_TASK_ID in
    0) BATCH_ACCUM=4; ARCH_ACCUM=1 ; LR=0.00085 ; ETA_MIN=8e-5; RESUME_FROM="/work/tesi_tsommariva/experiments/AaITERATIVE/SingleBlock_CustomInit_BatchAccumulationSteps:4_ArchAccumulationSteps:1_lr:0.00085/block7/cifar100_nerf_best.pth";;
    1) BATCH_ACCUM=1; ARCH_ACCUM=1 ; LR=3e-4 ; ETA_MIN=3e-5; RESUME_FROM="/work/tesi_tsommariva/experiments/AaITERATIVE/SingleBlock_CustomInit_BatchAccumulationSteps:1_ArchAccumulationSteps:1_lr:3e-4/block7/cifar100_nerf_best.pth";;
    *) echo "Invalid array task ID"; exit 1 ;;
esac

# Define experiment name
EXP_NAME="SingleBlock_CustomInit_BatchAccumulationSteps:${BATCH_ACCUM}_ArchAccumulationSteps:${ARCH_ACCUM}_lr:${LR}"

export PYTHONPATH=/homes/tsommariva/neumeta:$PYTHONPATH
python3 /homes/tsommariva/neumeta/neumeta/train_cifar100_iterative.py --config /homes/tsommariva/neumeta/neumeta/config/cifar100/Cifar100_resnet56_myConf_Iterative.yaml \
    --experiment.batch_accumulation_steps=$BATCH_ACCUM \
    --experiment.arch_accumulation_steps=$ARCH_ACCUM \
    --training.learning_rate=$LR \
    --training.eta_min=$ETA_MIN \
    --experiment.name="$EXP_NAME" \
    --resume_from="$RESUME_FROM"
    #--resume_from "experiments/AaITERATIVE/8Blocks_300e_4AccumulationSteps_bottomUp:True_Iterative:True_FINISHfromOldBlock2_150e/block2/cifar100_nerf_best.pth" \