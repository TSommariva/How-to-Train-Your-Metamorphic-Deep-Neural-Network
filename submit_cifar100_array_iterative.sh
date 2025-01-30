#!/bin/bash

#SBATCH --job-name=HybridTraining           
#SBATCH --output=log/HybridTraining/cvBackboneLR_%A_%a.out
#SBATCH  --error=log/HybridTraining/cvBackboneLR_%A_%a.err
#SBATCH --time=24:00:00                         
#SBATCH --constraint="gpu_A40_48G|gpu_RTX6000_24G|gpu_RTXA5000_24G" #|gpu_RTX5000_16G"
#SBATCH --array=0-3
#SBATCH --gres=gpu:1                            
#SBATCH --cpus-per-task=2

#SBATCH --account=tesi_tsommariva               
#SBATCH --partition=all_usr_prod                
#SBATCH --mail-type=ALL,ARRAY_TASKS                    
#SBATCH --mail-user=ts.slurm@gmail.com          

. /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate neumeta

# Define accumulation steps configurations
case $SLURM_ARRAY_TASK_ID in
    0)LR=6e-6;;
    1)LR=1e-6;;
    2)LR=3e-6;;
    3)LR=1e-5;;
    *) echo "Invalid array task ID"; exit 1 ;;
esac

export PYTHONPATH=/homes/tsommariva/neumeta:$PYTHONPATH
python3 /homes/tsommariva/neumeta/neumeta/train_cifar100_iterative.py --config /homes/tsommariva/neumeta/neumeta/config/cifar100/Cifar100_resnet56_myConf_Iterative.yaml \
    --training.backbone_learning_rate=$LR \
    #--resume_from "$RS"