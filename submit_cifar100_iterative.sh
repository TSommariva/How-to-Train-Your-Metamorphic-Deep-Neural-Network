#!/bin/bash

#SBATCH --job-name=iterTdFull            # Set the job name (optional, default is "slurm-[jobid]")
#SBATCH --output=log/AaITERATIVE/8Blocks_80e_3AccumulationSteps_TopDown_Full.out               # Redirect output to this file (%j will expand to jobID, optional, default is "slurm-[jobid].out")
#SBATCH --error=log/AaITERATIVE/8Blocks_80e_3AccumulationSteps_TopDown_Full.err                 # Redirect errors to this file (optional, default is the same as --output)
#SBATCH --time=24:00:00                # Set a limit on the total run time (mandatory if there's a system-wide default time limit)
#SBATCH --account=tesi_tsommariva
#SBATCH --partition=all_usr_prod       # Specify the partition/queue to submit to (optional, default depends on the system configuration)
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1                     # Total number of tasks across all nodes (optional, default is 1)
#SBATCH --nodes=1                      # Number of nodes to allocate (optional, default is 1)
#SBATCH --ntasks-per-node=1            # Number of tasks to run per node (optional, default is to divide tasks evenly)
#SBATCH --cpus-per-task=1              # Number of CPUs to allocate per task (optional, default is 1)
##SBATCH --mem=1000                     # Memory per node (in MB, optional, default is system specific)
#SBATCH --mail-type=END,FAIL           # Mail events (NONE, BEGIN, END, FAIL, ALL) (optional, default is NONE)
#SBATCH --mail-user=ts.slurm@gmail.com  # Where to send the mail (mandatory if mail-type is specified)
#SBATCH --constraint="gpu_A40_48G|gpu_RTX6000_24G|gpu_RTXA5000_24G|gpu_RTX5000_16G"

. /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate neumeta
# Command to execute Python program

export PYTHONPATH=/homes/tsommariva/neumeta:$PYTHONPATH
python3 /homes/tsommariva/neumeta/neumeta/train_cifar100.py --config /homes/tsommariva/neumeta/neumeta/config/cifar100/Cifar100_resnet56_myConf_Iterative.yaml