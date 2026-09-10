#!/bin/bash
# Brain MoE-PINN All-Node Launcher via SSH
# Usage: bash launch_all_nodes.sh <hosts_file> [train_args...]
#
# This script SSH's into each node and runs brain_moe_launch.sh
# with the appropriate NODE_RANK.

set -e

HOSTS_FILE=${1:-"hosts.txt"}
shift || true

if [ ! -f "$HOSTS_FILE" ]; then
    echo "ERROR: Hosts file not found: $HOSTS_FILE"
    exit 1
fi

NUM_NODES=$(wc -l < "$HOSTS_FILE")
GPUS_PER_NODE=4
MASTER_ADDR=$(head -n 1 "$HOSTS_FILE")
MASTER_PORT=${MASTER_PORT:-29500}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

echo "============================================"
echo "Brain MoE-PINN All-Node SSH Launcher"
echo "============================================"
echo "NUM_NODES:    $NUM_NODES"
echo "GPUS_PER_NODE:$GPUS_PER_NODE"
echo "MASTER_ADDR:  $MASTER_ADDR"
echo "MASTER_PORT:  $MASTER_PORT"
echo "PROJECT_DIR:  $PROJECT_DIR"
echo "============================================"

rank=0
while IFS= read -r host; do
    [ -z "$host" ] && continue
    echo "Launching on $host (rank=$rank)..."
    ssh -n -f "$host" \
        "cd $PROJECT_DIR && \
         export NODE_RANK=$rank && \
         export MASTER_ADDR=$MASTER_ADDR && \
         export MASTER_PORT=$MASTER_PORT && \
         bash brain_moe_launch.sh $NUM_NODES $HOSTS_FILE $@" \
        2>&1 | sed "s/^/[$host] /" &
    rank=$((rank + 1))
done < "$HOSTS_FILE"

echo ""
echo "All nodes launched. Waiting for jobs..."
wait
echo "All nodes finished."
