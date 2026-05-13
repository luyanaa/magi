#!/bin/bash
# Brain MoE-PINN Multi-Node Training Launch Script (SLURM)
# Usage: sbatch --nodes=16 --ntasks-per-node=4 --gres=gpu:4 brain_moe_multinode.sh [args...]
#
# For 64xV100 cluster: 16 nodes x 4 GPUs = 64 total GPUs
# Topology: TP=4, EP=4 (co-located on same NVLink node), DP=16

#SBATCH --job-name=brain_moe_pinn
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --partition=compute
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --exclusive

set -e

NNODES=${SLURM_JOB_NUM_NODES:-1}
GPUS_PER_NODE=4
NTASKS=$((NNODES * GPUS_PER_NODE))

echo "============================================"
echo "Brain MoE-PINN SLURM Multi-Node Training"
echo "============================================"
echo "Job ID:          $SLURM_JOB_ID"
echo "NNODES:          $NNODES"
echo "GPUS_PER_NODE:   $GPUS_PER_NODE"
echo "TOTAL_GPUS:      $NTASKS"
echo "NODELIST:        $SLURM_JOB_NODELIST"
echo "============================================"

# Create logs directory
mkdir -p logs

# Master node setup
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=${MASTER_PORT:-29500}
export WORLD_SIZE=$NTASKS

echo "MASTER_ADDR:     $MASTER_ADDR"
echo "MASTER_PORT:     $MASTER_PORT"
echo "WORLD_SIZE:      $WORLD_SIZE"

# NCCL tuning for multi-node V100 cluster (SXM2 NVLink + IB)
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-16}
export NCCL_IB_RETRY_CNT=${NCCL_IB_RETRY_CNT:-7}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-2}
export NCCL_NSOCKS_PERTHREAD=${NCCL_NSOCKS_PERTHREAD:-4}
export NCCL_SOCKET_NTHREADS=${NCCL_SOCKET_NTHREADS:-2}
export NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-4}
export NCCL_TREE_THRESHOLD=${NCCL_TREE_THRESHOLD:-0}
export NCCL_ALGO=${NCCL_ALGO:-RING}

# PyTorch distributed tuning
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-0}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-INFO}

# CUDA settings
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}

# DeepSpeed settings
export DS_SKIP_CUDA_CHECK=${DS_SKIP_CUDA_CHECK:-0}

echo ""
echo "NCCL / Communication Settings:"
echo "  NCCL_DEBUG:            $NCCL_DEBUG"
echo "  NCCL_IB_TIMEOUT:       $NCCL_IB_TIMEOUT"
echo "  NCCL_IB_RETRY_CNT:     $NCCL_IB_RETRY_CNT"
echo "  NCCL_SHM_DISABLE:      $NCCL_SHM_DISABLE"
echo "  NCCL_NET_GDR_LEVEL:    $NCCL_NET_GDR_LEVEL"
echo "  NCCL_NSOCKS_PERTHREAD: $NCCL_NSOCKS_PERTHREAD"
echo "  NCCL_SOCKET_NTHREADS:  $NCCL_SOCKET_NTHREADS"
echo "  NCCL_MIN_NCHANNELS:    $NCCL_MIN_NCHANNELS"
echo "  NCCL_TREE_THRESHOLD:   $NCCL_TREE_THRESHOLD"
echo "  NCCL_ALGO:             $NCCL_ALGO"
echo "============================================"

# Module and environment setup (adjust for your cluster)
# module load cuda/11.8
# module load nccl/2.18
# source /path/to/venv/bin/activate

TRAIN_SCRIPT=${TRAIN_SCRIPT:-"scripts/train.py"}
DS_CONFIG=${DS_CONFIG:-"configs/ds_config_zero2_multinode.json"}

# Option 1: srun + torchrun (most portable)
echo ""
echo "Launching via srun + torch.distributed.run..."
srun --ntasks=$NTASKS \
    --ntasks-per-node=$GPUS_PER_NODE \
    --gpus-per-task=1 \
    python -m torch.distributed.run \
        --nnodes=$NNODES \
        --nproc_per_node=$GPUS_PER_NODE \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --node_rank=$SLURM_NODEID \
        --rdzv_id=brain_moe_pinn_${SLURM_JOB_ID} \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        $TRAIN_SCRIPT \
        --deepspeed_config $DS_CONFIG \
        --epochs 100 \
        --phase "-1,0,1,2,3" \
        "$@"

# Option 2: DeepSpeed launcher (alternative, uncomment if preferred)
# deepspeed --num_gpus=$GPUS_PER_NODE \
#     --num_nodes=$NNODES \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$MASTER_PORT \
#     --hostfile=<(scontrol show hostnames "$SLURM_JOB_NODELIST" | awk '{print $1 " slots=" ENVIRON["GPUS_PER_NODE"]}') \
#     $TRAIN_SCRIPT \
#     --deepspeed_config $DS_CONFIG \
#     --epochs 100 \
#     --phase "-1,0,1,2,3" \
#     "$@"
