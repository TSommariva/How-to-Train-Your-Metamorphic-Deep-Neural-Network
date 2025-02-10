#!/bin/bash

#SBATCH --job-name=TrainPrior           
#SBATCH --output=log/TrainPrior/TrainPrior_%A_%a.out
#SBATCH  --error=log/TrainPrior/TrainPrior_%A_%a.err
#SBATCH --time=3:00:00                         
#SBATCH --constraint="gpu_A40_48G|gpu_RTX6000_24G|gpu_RTXA5000_24G" #|gpu_RTX5000_16G|gpu_2080Ti_11G"
#SBATCH --array=0-5
#SBATCH --gres=gpu:1                            
#SBATCH --cpus-per-task=2

#SBATCH --account=tesi_tsommariva               
#SBATCH --partition=all_usr_prod                
#SBATCH --mail-type=ALL,ARRAY_TASKS                    
#SBATCH --mail-user=ts.slurm@gmail.com          

. /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate neumeta

# Define learning rates and batch sizes
declare -a learning_rates=(1e-3 8e-3 8e-4)
declare -a batch_sizes=(64 128)

# Calculate indices for the current combination
lr_index=$((SLURM_ARRAY_TASK_ID / 3))
bs_index=$((SLURM_ARRAY_TASK_ID % 2))

# Get the values for the current run
LR=${learning_rates[$lr_index]}
BS=${batch_sizes[$bs_index]}


export PYTHONPATH=/homes/tsommariva/neumeta:$PYTHONPATH
python3 neumeta/trainResnet56XL.py --config neumeta/config/cifar100/Cifar100_resnet56.yaml \
    --training.batch_size=$BS \
    --training.learning_rate=$LR 