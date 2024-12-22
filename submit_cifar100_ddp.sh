#!/bin/bash

#SBATCH --job-name=NeRF         
#SBATCH --output=log/AaDDP/8Blocks_600e_2AccumulationSteps_BottomUp_Full_resumeFormOldB2_e150.out
#SBATCH --error=log/AaDDP/8Blocks_6800e_2AccumulationSteps_BottomUp_Full_resumeFormOldB2_e150.err

#SBATCH --account=tesi_tsommariva
#SBATCH --partition=all_usr_prod       
#SBATCH --time=24:00:00                
#SBATCH --constraint="gpu_RTX5000_16G|gpu_A40_48G|gpu_RTX6000_24G|gpu_RTXA5000_24G"
#SBATCH --mail-type=END,FAIL           
#SBATCH --mail-user=ts.slurm@gmail.com 

#SBATCH --gres=gpu:2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2

#SBATCH --nodes=1
#SBATCH --cpus-per-task=2

GPUS_PER_NODE=2

echo "SLURM_NTASKS="$SLURM_NTASKS
NTASKS_PER_NODE=$((SLURM_NTASKS / SLURM_JOB_NUM_NODES))
echo "NTASKS_PER_NODE="$NTASKS_PER_NODE
export WORLD_SIZE=$((GPUS_PER_NODE * SLURM_NNODES))
echo "WORLD_SIZE=$WORLD_SIZE"

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=11111

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

#export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800

. /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate neumeta
export PYTHONPATH=/homes/tsommariva/neumeta:$PYTHONPATH


mpirun -np $WORLD_SIZE -x MASTER_ADDR=$MASTER_ADDR -x MASTER_PORT=$MASTER_PORT \
    --map-by numa \
    python3 neumeta/train_cifar100_ddp.py --config neumeta/config/cifar100/Cifar100_resnet56_myConf_ddp.yaml \
    --resume_from "experiments/AaITERATIVE/8Blocks_150e_4AccumulationSteps_bottomUp:True_Iterative:True_Full/block2/cifar100_nerf_best.pth"