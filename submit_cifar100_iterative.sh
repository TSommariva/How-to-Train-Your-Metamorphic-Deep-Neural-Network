#!/bin/bash

#SBATCH --job-name=FullSimMetamorphosis
#SBATCH --output=log/AaPaper/Full/FullSimultaneus_Finish_from2557785_%j.out
#SBATCH  --error=log/AaPaper/Full/FullSimultaneus_Finish_from2557785_%j.err
#SBATCH --time=24:00:00                         
#SBATCH --constraint="gpu_L40S_48G|gpu_A40_48G|gpu_RTXA5000_24G|gpu_RTX6000_24G|gpu_RTX5000_16G|gpu_2080Ti_11G"
#SBATCH --mem=24G

#SBATCH --gres=gpu:1                            
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

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
    --resume_from "/work/tesi_tsommariva/experiments/Paper/FULL_allLayersSim:True_Blocks1to8_v2/layer2/nerf_block7_best.pth" \
    --experiment.name='FULL_allLayersSim:True_Blocks1to8_from2557785' \
    #--experiment.all_layers_sim=False \
    #--hyper_model.type='resmlp' \
    #--experiment.test_path='/work/tesi_tsommariva/experiments/Paper/lastBlock_resmlp_100e_skipInit_StartBlock8_CustmInit:True_4AccumulationSteps' \
    #--experiment.name='LastBlock_Eval_NoDisentanglement' \
    #--experiment.test=True \
    #--model.metamorphic_block_type='resize' \
    #--experiment.batch_accumulation_steps=1 \
    #--training.cls_learning_rate=0.0 \
    #--experiment.num_epochs=350 \
    #--model.start_block=7 \
    #--hyper_model.type='resmlpDict'
    #--training.scheduler="warmup_const_cosine" \