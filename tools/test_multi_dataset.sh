#!/usr/bin/env bash
CONFIG=$1 # config file path
CHECKPOINT=$2 # checkpoint path
DATASET_NUM=$3 # number of dataset for test
for ((i=0; i<$DATASET_NUM; i++))
do
    python tools/test.py $CONFIG $CHECKPOINT --cfg-options test_dataloader.dataset.sub_data_index=$i
done
