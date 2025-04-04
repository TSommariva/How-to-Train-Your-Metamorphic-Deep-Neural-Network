#!/bin/bash

#SBATCH --job-name=NeuMeta           
#SBATCH --output=log/AaPaper/Ablation/LastBlock/Eval_%A_%a.out
#SBATCH  --error=log/AaPaper/Ablation/LastBlock/Eval_%A_%a.err
#SBATCH --time=2:00:00                         
#SBATCH --constraint="gpu_A40_48G|gpu_RTX6000_24G|gpu_RTXA5000_24G|gpu_RTX5000_16G"
#SBATCH --array=0-2
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
    0)path='/work/tesi_tsommariva/experiments/Paper/lastBlock_resmlpDict_100e_skipInit_StartBlock8_CustmInit:True_4AccumulationSteps';;
    1)path='/work/tesi_tsommariva/experiments/Paper/lastBlock_resmlpDict_100e_skipInit_StartBlock8_CustmInit:True_1AccumulationSteps';;
    2)path='/work/tesi_tsommariva/experiments/Paper/lastBlock_resmlp_100e_skipInit_StartBlock8_CustmInit:True_4AccumulationSteps';;
    *) echo "Invalid array task ID"; exit 1 ;;
esac
case $SLURM_ARRAY_TASK_ID in
    0)name='LastBlock_Eval_original';;
    1)name='LastBlock_Eval_NoAccumulation';;
    2)name='LastBlock_Eval_NoDisentanglement';;
    *) echo "Invalid array task ID"; exit 1 ;;
esac

export PYTHONPATH=/homes/tsommariva/neumeta:$PYTHONPATH
python3 /homes/tsommariva/neumeta/neumeta/train_cifar100_iterative.py --config /homes/tsommariva/neumeta/neumeta/config/cifar100/Cifar100_resnet56_myConf_Iterative.yaml \
    --experiment.test_path=$path \
    --experiment.name=$name \
    --experiment.test=True \