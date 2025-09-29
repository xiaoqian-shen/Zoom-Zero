full_model_path=$1
prefix=$2
data_path=$3

export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1

torchrun \
    --nnodes=$SLURM_NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --master_port=$MASTER_PORT \
    -m src_eval.eval_nextgqa \
    --model_path $full_model_path \
    --num_gpus $SLURM_GPUS_ON_NODE \
    --data_path $data_path \
    --prefix $prefix