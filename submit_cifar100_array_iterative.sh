#!/bin/bash

#SBATCH --job-name=SkipInit           
#SBATCH --output=log/SkipInit/DictNerf_SkipInit_cvwd_%A_%a.out
#SBATCH  --error=log/SkipInit/DictNerf_SkipInit_cvwd_%A_%a.err
#SBATCH --time=4:00:00                         
#SBATCH --constraint="gpu_A40_48G|gpu_RTX6000_24G|gpu_RTXA5000_24G|gpu_RTX5000_16G"
#SBATCH --array=0-3
#SBATCH --gres=gpu:1                            
#SBATCH --cpus-per-task=6

#SBATCH --account=tesi_tsommariva               
#SBATCH --partition=all_usr_prod                
#SBATCH --mail-type=ALL,ARRAY_TASKS                    
#SBATCH --mail-user=ts.slurm@gmail.com          

. /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate neumeta

# Define accumulation steps configurations
case $SLURM_ARRAY_TASK_ID in
    0)WD=1e-4 ;;
    1)WD=1e-3 ;;
    2)WD=1e-1 ;;
    3)WD=1e-5 ;;
    *) echo "Invalid array task ID"; exit 1 ;;
esac

export PYTHONPATH=/homes/tsommariva/neumeta:$PYTHONPATH
python3 /homes/tsommariva/neumeta/neumeta/train_cifar100_iterative.py --config /homes/tsommariva/neumeta/neumeta/config/cifar100/Cifar100_resnet56_myConf_Iterative.yaml \
    --training.cls_weight_decay=$WD \
    #--resume_from "$RESUME_FROM"