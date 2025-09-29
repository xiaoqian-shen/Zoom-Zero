full_model_path=$1
prefix=$2
data_path=$3
chunk_size=$4

export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1

echo 'num_gpus: '$((SLURM_GPUS_ON_NODE * SLURM_NNODES))

torchrun \
    --nnodes=$SLURM_NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --master_port=$MASTER_PORT \
    -m src_eval.eval_cgbench_zoom \
    --model_path $full_model_path \
    --num_gpus $((SLURM_GPUS_ON_NODE * SLURM_NNODES)) \
    --prefix $prefix \
    --data_path $data_path \
    --chunk_size $chunk_size