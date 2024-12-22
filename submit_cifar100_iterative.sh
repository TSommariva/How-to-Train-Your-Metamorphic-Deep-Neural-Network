#!/bin/bash

#SBATCH --job-name=ArcBatchAccumulation          
#SBATCH --output=log/AaITERATIVE/batch_accumulation_steps1_arch_accumulation_steps4_v2.out
#SBATCH --error=log/AaITERATIVE/batch_accumulation_steps1_arch_accumulation_steps4_v2.err 
#SBATCH --time=24:00:00                         
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
BATCH_ACCUM=1; ARCH_ACCUM=4; EPOCHS=50
EXP_NAME="batch_accumulation_steps:${BATCH_ACCUM}_arch_accumulation_steps:${ARCH_ACCUM}__epchs:${EPOCHS}fromBeginning"

export PYTHONPATH=/homes/tsommariva/neumeta:$PYTHONPATH
python3 /homes/tsommariva/neumeta/neumeta/train_cifar100.py --config /homes/tsommariva/neumeta/neumeta/config/cifar100/Cifar100_resnet56_myConf_Iterative.yaml \
    --experiment.batch_accumulation_steps=$BATCH_ACCUM \
    --experiment.arch_accumulation_steps=$ARCH_ACCUM \
    --experiment.num_epochs=$EPOCHS \
    --experiment.name="$EXP_NAME" 
    #--resume_from "experiments/AaITERATIVE/8Blocks_300e_4AccumulationSteps_bottomUp:True_Iterative:True_FINISHfromOldBlock2_150e/block2/cifar100_nerf_best.pth" \