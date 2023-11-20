#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.
#  PORT=10040 bash tools/dist_train.sh configs/product/stliu/debug.py  1 --work-dir work_dirs/pair_hand3d_old/test
# PORT=10015 bash tools/dist_train.sh configs/product/stliu/debug.py   8 --amp  --auto-scale-lr --work-dir work_dirs/pair_hand3d_old/test
# PORT=10006 bash tools/dist_train.sh configs/product/stliu/stliu_046_fit_xyz_3dloss_svdpose.py   8 --amp  --auto-scale-lr --work-dir work_dirs/pair_hand3d_new/fit_xyz_3dloss_svdpose

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$GPUS \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}
