#!/bin/bash

export SLURM_NTASKS=1
export HYDRA_FULL_ERROR=1
export PYTHONPYCACHEPREFIX=$TMPDIR/pycache
export CUDA_LAUNCH_BLOCKING=1

# Prevent inheriting distributed settings from other shells/jobs.
unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT NODE_RANK
unset TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT TORCHELASTIC_MAX_RESTARTS
unset RDZV_BACKEND RDZV_ENDPOINT RDZV_ID

if [ -n "$NUM_GPUS" ]; then
    NUM_GPUS="$NUM_GPUS"
elif [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
fi

# Unique communication endpoint per run so unrelated torchrun jobs never meet.
MASTER_PORT=${MASTER_PORT:-$((10000 + (RANDOM % 50000)))}

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Launching with NUM_GPUS=${NUM_GPUS} MASTER_PORT=${MASTER_PORT}"

mkdir -p $TMPDIR

# Use torchrun instead of python
torchrun \
    --nnodes 1 \
    --node_rank 0 \
    --nproc_per_node "$NUM_GPUS" \
    --master_addr 127.0.0.1 \
    --master_port "$MASTER_PORT" \
    main.py \
    --config-dir=configs \
    --config-name=train-toy-dataset.yaml