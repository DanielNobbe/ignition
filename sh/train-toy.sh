#!/bin/bash

export SLURM_NTASKS=1
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export NCCL_P2P_LEVEL=NVL  # omg it works

# Use torchrun instead of python
torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=10052 \
    main.py \
    --config-dir=configs \
    --config-name=toy_dataset.yaml