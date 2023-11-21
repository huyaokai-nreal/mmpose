#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.
# PORT=10020 bash tools/dist_test.sh  configs/product/stliu/debug.py /data/stliu/mmpose/work_dirs/pair_hand3d_new/fit_xyz_3dloss_svd/best_all_mpjpe_epoch_140.pth 4 --work-dir work_dirs/pair_hand3d_old/test

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$GPUS \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    ${@:4}
