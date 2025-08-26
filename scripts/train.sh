#!/bin/bash

devices=$1
config=$2
resume=$3
nproc_per_node=$(echo ${devices%%,} | grep -o "," | wc -l)
to_be_distributed=`echo ${nproc_per_node} | awk '{if($e > 0) print "True"; else print "False";}'`

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=./
export TOKENIZERS_PARALLELISM=false
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

if [ ${to_be_distributed} == "True" ]
then
    echo "Multi-GPU mode received..."
    CUDA_VISIBLE_DEVICES=${devices} \
    torchrun --nnodes=1 --nproc_per_node=$((nproc_per_node+1)) --master_port=$((29061+${4:-11})) --node_rank=0 \
    scripts/train.py --config=${config} --resume=${resume}
else
echo "Single-GPU mode received..."
CUDA_VISIBLE_DEVICES=$devices python scripts/train.py --config=${config} --resume=${resume}
fi
