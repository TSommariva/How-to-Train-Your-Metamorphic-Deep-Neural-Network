#!/bin/bash

#SBATCH --job-name=ArcBatchAccumulation           
#SBATCH --output=log/AaITERATIVE/arc_batch_accumulation_resumeFromBlock3_%a.out
#SBATCH --error=log/AaITERATIVE/arc_batch_accumulation_resumeFromBlock3_%a.err
#SBATCH --time=12:00:00                         
#SBATCH --constraint="gpu_A40_48G|gpu_RTX6000_24G|gpu_RTXA5000_24G|gpu_RTX5000_16G"
#SBATCH --array=0-2
#SBATCH --gres=gpu:1                            
#SBATCH --cpus-per-task=2

#SBATCH --account=tesi_tsommariva               
#SBATCH --partition=all_usr_prod                
#SBATCH --mail-type=END,FAIL                    
#SBATCH --mail-user=ts.slurm@gmail.com          

. /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate neumeta

# Define accumulation steps configurations
case $SLURM_ARRAY_TASK_ID in
    0) BATCH_ACCUM=1; ARCH_ACCUM=4 ;;
    1) BATCH_ACCUM=2; ARCH_ACCUM=2 ;;
    2) BATCH_ACCUM=4; ARCH_ACCUM=1 ;;
    *) echo "Invalid array task ID"; exit 1 ;;
esac

# Define experiment name
EXP_NAME="batch_accumulation_steps:${BATCH_ACCUM}_arch_accumulation_steps:${ARCH_ACCUM}"

export PYTHONPATH=/homes/tsommariva/neumeta:$PYTHONPATH
python3 /homes/tsommariva/neumeta/neumeta/train_cifar100.py --config /homes/tsommariva/neumeta/neumeta/config/cifar100/Cifar100_resnet56_myConf_Iterative.yaml \
    --resume_from "experiments/AaITERATIVE/8Blocks_300e_4AccumulationSteps_bottomUp:True_Iterative:True_FINISHfromOldBlock2_150e/block2/cifar100_nerf_best.pth" \
    --experiment.batch_accumulation_steps=$BATCH_ACCUM \
    --experiment.arch_accumulation_steps=$ARCH_ACCUM \
    --experiment.name="$EXP_NAME"