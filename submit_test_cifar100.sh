#!/bin/bash

#SBATCH --job-name=NeRFtest
#SBATCH --output=log/test/Test_%j.out
#SBATCH  --error=log/test/Test_%j.err
#SBATCH --time=4:00:00                         

#SBATCH --gres=gpu:1                            
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

#SBATCH --account=tesi_tsommariva               
#SBATCH --partition=all_usr_prod                
#SBATCH --mail-type=ALL                   
#SBATCH --mail-user=ts.slurm@gmail.com          

. /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate neumeta
# Command to execute Python program
export PYTHONPATH=/homes/tsommariva/neumeta:$PYTHONPATH
python3 /homes/tsommariva/neumeta/neumeta/test_cifar100.py --config /homes/tsommariva/neumeta/neumeta/config/cifar100/Cifar100_resnet56_myConf_Test.yaml \
    --resume_from "/work/tesi_tsommariva/experiments/AaITERATIVE/skipInit_clsLr0.001_resmlpDict_Scalar0_4kernelGroups_4AccumulationSteps_50e_lr0.00085_bs128_warmup_cosine20/cifar100_nerf_last.pth" \