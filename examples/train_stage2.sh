set -x

OUTPUT_DIR=$1
MODEL_PATH=$2
EXP_NAME=$(basename ${OUTPUT_DIR})

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0
export WANDB_RESUME="allow"
export WANDB_RUN_ID=$EXP_NAME
export CUDA_LAUNCH_BLOCKING=1

python3 -m verl.trainer.main \
    config=examples/grpo_example.yaml \
    data.train_files=examples/data_configs/stage2.yaml \
    data.val_files=examples/data_configs/eval.yaml \
    data.max_prompt_length=10000 \
    data.max_response_length=512 \
    data.rollout_batch_size=64 \
    worker.rollout.max_num_batched_tokens=12000 \
    worker.actor.global_batch_size=$((SLURM_NNODES * 8)) \
    worker.actor.global_batch_size_per_device=8 \
    worker.actor.entropy_coeff=0 \
    worker.actor.kl_loss_coef=1e-2 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=4 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.n=8 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${SLURM_NNODES} \
    trainer.val_generations_to_log=10 \
    trainer.save_limit=3 \
    trainer.save_freq=1 \
    trainer.val_freq=5 \
    trainer.val_before_train=false \
    trainer.logger=[\"console\",\"wandb\"] \
    trainer.save_checkpoint_path=${OUTPUT_DIR} \
    trainer.load_checkpoint_path=${OUTPUT_DIR} \
    data.min_pixels=$((16*28*28)) \
    data.max_pixels=$((8192*28*28)) \
    worker.reward.compute_score=tvg_zoom \
    algorithm.adv_estimator=grpo_select \
    worker.reward.reason_reward=true \
    algorithm.zoom=true \
