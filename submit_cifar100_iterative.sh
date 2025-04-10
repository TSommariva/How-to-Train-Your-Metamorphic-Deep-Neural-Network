#!/bin/bash

#SBATCH --job-name=NoGradAcc
#SBATCH --output=log/AaPaper/XL/NoGradAcc_Finish_%j.out
#SBATCH  --error=log/AaPaper/XL/NoGradAcc_Finish_%j.err
#SBATCH --time=24:00:00                         
#SBATCH --constraint="gpu_A40_48G|gpu_L40S_48G|gpu_RTXA5000_24G|gpu_RTX6000_24G" #|gpu_RTX5000_16G" #|gpu_2080Ti_11G"
#SBATCH --mem=48G

#SBATCH --gres=gpu:1                            
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

#SBATCH --account=tesi_tsommariva               
#SBATCH --partition=boost_usr_prod                
#SBATCH --mail-type=ALL                   
#SBATCH --mail-user=ts.slurm@gmail.com          


. /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate neumeta
# Command to execute Python program

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/homes/tsommariva/neumeta:$PYTHONPATH

python3 /homes/tsommariva/neumeta/neumeta/train_cifar100_iterative.py --config neumeta/config/cifar100/Cifar100_resnet56_myConf_Iterative.yaml \
    --resume_from "/work/tesi_tsommariva/experiments/Paper/RXL_lastBlock_resmlpDict_50e_skipInit_StartBlock1_CustmInit:True_1AccumulationSteps/nerf_last.pth" \
    --experiment.batch_accumulation_steps=1 \
    #--experiment.custom_init=False \
    #--hyper_model.type='resmlp' \
    #--training.alpha_learning_rate=0 \
    #--model.metamorphic_block_type='resize' \
    #--model.start_block=7 \
    #--experiment.num_epochs=350 \
    #--training.cls_learning_rate=0 \
    #--training.learning_rate=1e-4 \
