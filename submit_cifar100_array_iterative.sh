#!/bin/bash

#SBATCH --job-name=DictNeRF           
#SBATCH --output=log/Debug/DictNerf_SGD_%A_%a.out
#SBATCH  --error=log/Debug/DictNerf_SGD_%A_%a.err
#SBATCH --time=1:00:00                         
#SBATCH --constraint="gpu_A40_48G|gpu_RTX6000_24G|gpu_RTXA5000_24G|gpu_RTX5000_16G"
#SBATCH --array=0-2
#SBATCH --gres=gpu:1                            
#SBATCH --cpus-per-task=2

#SBATCH --account=tesi_tsommariva               
#SBATCH --partition=all_usr_prod                
#SBATCH --mail-type=BEGIN,END,FAIL                    
#SBATCH --mail-user=ts.slurm@gmail.com          

. /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate neumeta

# Define accumulation steps configurations
case $SLURM_ARRAY_TASK_ID in
    0) LR=1e-1 ; ETA_MIN=1e-2; OPTIM='sgd';;
    1) LR=1e-2 ; ETA_MIN=1e-3 ; OPTIM='sgd';;
    2) LR=1e-3 ; ETA_MIN=1e-4 ; OPTIM='sgd';;
    *) echo "Invalid array task ID"; exit 1 ;;
esac

export PYTHONPATH=/homes/tsommariva/neumeta:$PYTHONPATH
python3 /homes/tsommariva/neumeta/neumeta/train_cifar100_iterative.py --config /homes/tsommariva/neumeta/neumeta/config/cifar100/Cifar100_resnet56_myConf_Iterative.yaml \
    --training.learning_rate=$LR \
    --training.eta_min=$ETA_MIN \
    --training.optimizer=$OPTIM
    #--resume_from "$RESUME_FROM"