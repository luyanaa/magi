#!/bin/bash
# Brain MoE-PINN Direct Multi-Node Launch Script
# Usage: bash brain_moe_launch.sh <num_nodes> <hosts_file> [additional_args]
# Example: bash brain_moe_launch.sh 16 hosts.txt
#
# For 64xV100 cluster: 16 nodes x 4 GPUs = 64 total GPUs
# Topology: TP=4, EP=4 (co-located on same NVLink node), DP=16
#
# Prerequisites:
#   - Passwordless SSH between all nodes
#   - Same Python environment on all nodes
#   - hosts.txt contains one hostname per line (first = master)

set -e

NUM_NODES=${1:-16}
GPUS_PER_NODE=4
TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))

HOSTS_FILE=${2:-"hosts.txt"}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-"train.py"}
DS_CONFIG=${DS_CONFIG:-"configs/ds_config_zero2_multinode.json"}

if [ ! -f "$HOSTS_FILE" ]; then
    echo "ERROR: Hosts file not found: $HOSTS_FILE"
    echo "Usage: bash brain_moe_launch.sh <num_nodes> <hosts_file>"
    echo ""
    echo "Create hosts.txt with one hostname per line, e.g.:"
    echo "  node01"
    echo "  node02"
    echo "  ..."
    exit 1
fi

MASTER_ADDR=$(head -n 1 "$HOSTS_FILE")
MASTER_PORT=${MASTER_PORT:-29500}

# NCCL tuning for multi-node V100 cluster
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-16}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-2}
export NCCL_NSOCKS_PERTHREAD=${NCCL_NSOCKS_PERTHREAD:-4}
export NCCL_SOCKET_NTHREADS=${NCCL_SOCKET_NTHREADS:-2}
export NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-4}
export NCCL_TREE_THRESHOLD=${NCCL_TREE_THRESHOLD:-0}

# PyTorch distributed tuning
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-0}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-INFO}

# CUDA / memory settings
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}

# DeepSpeed settings
export DS_SKIP_CUDA_CHECK=${DS_SKIP_CUDA_CHECK:-0}

echo "============================================"
echo "Brain MoE-PINN Multi-Node Training Launch"
echo "============================================"
echo "NUM_NODES:       $NUM_NODES"
echo "GPUS_PER_NODE:   $GPUS_PER_NODE"
echo "TOTAL_GPUS:      $TOTAL_GPUS"
echo "MASTER_ADDR:     $MASTER_ADDR"
echo "MASTER_PORT:     $MASTER_PORT"
echo "HOSTS_FILE:      $HOSTS_FILE"
echo "TRAIN_SCRIPT:    $TRAIN_SCRIPT"
echo "DS_CONFIG:       $DS_CONFIG"
echo "============================================"
echo "NCCL Settings:"
echo "  NCCL_DEBUG:              $NCCL_DEBUG"
echo "  NCCL_IB_TIMEOUT:         $NCCL_IB_TIMEOUT"
echo "  NCCL_SHM_DISABLE:        $NCCL_SHM_DISABLE"
echo "  NCCL_NET_GDR_LEVEL:      $NCCL_NET_GDR_LEVEL"
echo "  NCCL_NSOCKS_PERTHREAD:   $NCCL_NSOCKS_PERTHREAD"
echo "  NCCL_SOCKET_NTHREADS:    $NCCL_SOCKET_NTHREADS"
echo "  NCCL_MIN_NCHANNELS:      $NCCL_MIN_NCHANNELS"
echo "  NCCL_TREE_THRESHOLD:     $NCCL_TREE_THRESHOLD"
echo "============================================"

# Option 1: torch.distributed.run (preferred for direct launch)
# Only launch from master node; other nodes join via init_method
NODE_RANK=${NODE_RANK:-0}

echo ""
echo "Launching with torchrun (node_rank=$NODE_RANK)..."
echo "Run this script on EACH node with NODE_RANK set appropriately."
echo "Or use the launcher wrapper: launch_all_nodes.sh"
echo ""

python -m torch.distributed.run \
    --nnodes=$NUM_NODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank=$NODE_RANK \
    --rdzv_id=brain_moe_pinn_$$ \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --max_restarts=3 \
    --monitor_interval=5 \
    $TRAIN_SCRIPT \
    --deepspeed_config $DS_CONFIG \
    --epochs 100 \
    --phase "-1,1,2,3" \
    ${@:3}
