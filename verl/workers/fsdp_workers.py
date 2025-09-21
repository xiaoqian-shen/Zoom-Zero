# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The main entry point to run the PPO algorithm
"""

import os
from typing import Literal, Optional
import re
import ast
import numpy as np

import torch
import torch.distributed as dist
from accelerate import init_empty_weights
from codetiming import Timer
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tensordict import TensorDict
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    GenerationConfig,
    PreTrainedModel,
)
from transformers.modeling_utils import no_init_weights

from ..models.monkey_patch import apply_ulysses_patch
from ..protocol import DataProto
from ..single_controller.base import Worker
from ..single_controller.base.decorator import Dispatch, register
from ..utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from ..utils.flops_counter import FlopsCounter
from ..utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_fn,
    load_fsdp_model,
    load_fsdp_optimizer,
    offload_fsdp_model,
    offload_fsdp_optimizer,
)
from ..utils.model_utils import print_gpu_memory_usage, print_model_size
from ..utils.tokenizer import get_processor, get_tokenizer
from ..utils.torch_dtypes import PrecisionType
from ..utils.torch_functional import get_constant_schedule_with_warmup, postprocess_data, get_cosine_schedule_with_warmup
from .actor import DataParallelPPOActor
from .config import FSDPConfig, ModelConfig, OptimConfig, WorkerConfig
from .critic import DataParallelPPOCritic
from .rollout.vllm_rollout import vLLMRollout
from .sharding_manager import FSDPVLLMShardingManager
from .sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from ..models.transformers.qwen2_vl import get_rope_index
from ..utils.reward_score.tvg import is_valid_two_d_list_format
from ..utils.dataset_video import resize_video


class FSDPWorker(Worker):
    def __init__(
        self,
        config: WorkerConfig,
        role: Literal["actor", "critic", "rollout", "ref", "actor_rollout", "actor_rollout_ref"],
    ):
        super().__init__()
        self.config = config

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        # build device mesh for FSDP
        # TODO: support FSDP hybrid shard for larger model
        local_rank = os.getenv("LOCAL_RANK")
        world_size = dist.get_world_size()
        self.device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
        torch.cuda.set_device(f"cuda:{local_rank}")

        # build device mesh for Ulysses Sequence Parallel
        self.ulysses_sequence_parallel_size = self.config.actor.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(world_size // self.ulysses_sequence_parallel_size, self.ulysses_sequence_parallel_size),
                mesh_dim_names=["dp", "sp"],
            )
        else:
            self.ulysses_device_mesh = None

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        self.role = role
        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_critic = self.role == "critic"
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        self._use_param_offload = False
        self._use_optimizer_offload = False
        if self._is_actor:
            self._use_param_offload = self.config.actor.offload.offload_params
            self._use_optimizer_offload = self.config.actor.offload.offload_optimizer
        elif self._is_critic:
            self._use_param_offload = self.config.critic.offload.offload_params
            self._use_optimizer_offload = self.config.critic.offload.offload_optimizer
        elif self._is_ref:  # NOTE: it seems that manual offload is slowly than FSDP offload
            self._use_param_offload = self.config.ref.offload.offload_params

        # normalize config
        if self._is_actor:
            if self.config.rollout.n > 1:
                self.config.actor.global_batch_size *= self.config.rollout.n
                self.print_rank0(f"Use global batch size {self.config.actor.global_batch_size}.")

            self.config.actor.global_batch_size_per_device = (
                self.config.actor.global_batch_size // self.device_mesh.shape[0] * self.ulysses_sequence_parallel_size
            )
            if (
                self.config.actor.global_batch_size_per_device
                % self.config.actor.micro_batch_size_per_device_for_update
                != 0
            ):
                raise ValueError("Global batch size should be divisible by the micro batch size.")

            if (
                self.config.actor.fsdp.enable_cpu_offload
                and self.config.actor.global_batch_size_per_device
                != self.config.actor.micro_batch_size_per_device_for_update
            ):
                raise ValueError("Cannot use FSDP's CPU offload when gradient accumulation is enabled.")

        elif self._is_critic:
            if self.config.rollout.n > 1:
                self.config.critic.global_batch_size *= self.config.rollout.n
                self.print_rank0(f"Use global batch size {self.config.critic.global_batch_size}.")

            self.config.critic.global_batch_size_per_device = (
                self.config.critic.global_batch_size // self.device_mesh.shape[0] * self.ulysses_sequence_parallel_size
            )
            if (
                self.config.critic.global_batch_size_per_device
                % self.config.critic.micro_batch_size_per_device_for_update
                != 0
            ):
                raise ValueError("Global batch size should be divisible by the micro batch size.")

            if (
                self.config.critic.fsdp.enable_cpu_offload
                and self.config.critic.global_batch_size_per_device
                != self.config.critic.micro_batch_size_per_device_for_update
            ):
                raise ValueError("Cannot use FSDP's CPU offload when gradient accumulation is enabled.")

    def _build_model_optimizer(
        self,
        model_config: ModelConfig,
        fsdp_config: FSDPConfig,
        optim_config: Optional[OptimConfig],
        padding_free: bool = False,
    ) -> None:
        self.tokenizer = get_tokenizer(
            model_config.tokenizer_path,
            trust_remote_code=model_config.trust_remote_code,
            use_fast=True,
        )
        self.processor = get_processor(
            model_config.tokenizer_path,
            trust_remote_code=model_config.trust_remote_code,
            use_fast=True,
        )
        self.model_config = AutoConfig.from_pretrained(
            model_config.model_path,
            trust_remote_code=model_config.trust_remote_code,
            # bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_config.override_config,
        )

        try:
            self.generation_config = GenerationConfig.from_pretrained(model_config.model_path)
        except Exception:
            self.generation_config = GenerationConfig.from_model_config(self.model_config)

        self.print_rank0(f"Model config: {self.model_config}")

        if padding_free:
            apply_ulysses_patch(self.model_config.model_type)
            self.print_rank0("Ulysses patch applied!")

        if fsdp_config.torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor or self._is_critic else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(fsdp_config.torch_dtype)

        if self._is_critic:
            auto_class = AutoModelForTokenClassification
        elif type(self.model_config) in AutoModelForVision2Seq._model_mapping.keys():
            auto_class = AutoModelForVision2Seq
        else:
            auto_class = AutoModelForCausalLM

        if (not fsdp_config.enable_rank0_init) or self.rank == 0:
            model = auto_class.from_pretrained(
                model_config.model_path,
                config=self.model_config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                device_map="cpu" if fsdp_config.enable_rank0_init else "cuda",
                low_cpu_mem_usage=True,
                trust_remote_code=model_config.trust_remote_code,
            )
        else:
            with no_init_weights(), init_empty_weights():
                model = auto_class.from_config(
                    self.model_config,
                    torch_dtype=torch_dtype,
                    attn_implementation="flash_attention_2",
                    trust_remote_code=model_config.trust_remote_code,
                )

        assert isinstance(model, PreTrainedModel)  # lint
        model.tie_weights()  # avoid hanging
        model = model.to(torch_dtype)
        if model_config.enable_gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if not (self._is_actor or self._is_critic):
            model.requires_grad_(False)

        if model_config.freeze_vision_tower:
            if hasattr(model, "visual"):
                model.visual.requires_grad_(False)
                fsdp_config.use_orig_params = True
                self.print_rank0("Vision tower is set to not trainable.")
            else:
                self.print_rank0("No vision tower found.")

        dist.barrier()
        if self.rank == 0:
            print_model_size(model)

        print_gpu_memory_usage("After init from huggingface model")
        mixed_precision = MixedPrecision(
            param_dtype=PrecisionType.to_dtype(fsdp_config.mp_param_dtype),
            reduce_dtype=PrecisionType.to_dtype(fsdp_config.mp_reduce_dtype),
            buffer_dtype=PrecisionType.to_dtype(fsdp_config.mp_buffer_dtype),
        )
        auto_wrap_policy = get_fsdp_wrap_policy(model)
        self.print_rank0(f"FSDP wrap policy: {auto_wrap_policy}.")

        if fsdp_config.enable_full_shard:
            sharding_strategy = ShardingStrategy.FULL_SHARD
        else:
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP

        if fsdp_config.enable_cpu_offload:
            cpu_offload = CPUOffload(offload_params=True)
        else:
            cpu_offload = None

        if fsdp_config.enable_rank0_init:
            sync_module_states = True
            param_init_fn = get_init_fn(model, device="cuda") if self.rank != 0 else None
        else:
            sync_module_states = False
            param_init_fn = None

        self.fsdp_module = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            param_init_fn=param_init_fn,
            device_id=torch.cuda.current_device(),
            sync_module_states=sync_module_states,
            forward_prefetch=False,
            use_orig_params=fsdp_config.use_orig_params,
            device_mesh=self.device_mesh,
        )
        print_gpu_memory_usage("After actor FSDP init")

        if self._is_actor or self._is_critic:
            self.optimizer = torch.optim.AdamW(
                self.fsdp_module.parameters(),
                lr=optim_config.lr,
                betas=optim_config.betas,
                weight_decay=optim_config.weight_decay,
            )
            num_warmup_steps = int(optim_config.lr_warmup_ratio * optim_config.training_steps)
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=10
            )
        else:
            self.optimizer, self.lr_scheduler = None, None

        print_gpu_memory_usage("After actor optimizer init")

    def _build_rollout(self) -> None:
        # TODO(sgm): support FSDP hybrid shard for larger model
        tp_size = self.config.rollout.tensor_parallel_size
        dp_size = self.world_size // tp_size
        assert self.world_size % tp_size == 0, (
            f"rollout world_size: {self.world_size} is not divisible by tp_size: {tp_size}"
        )
        rollout_device_mesh = init_device_mesh("cuda", mesh_shape=(dp_size, tp_size), mesh_dim_names=["dp", "tp"])
        print_gpu_memory_usage("Before building vllm rollout")
        self.rollout = vLLMRollout(
            model_path=self.config.actor.model.model_path,
            config=self.config.rollout,
            tokenizer=self.tokenizer,
        )
        print_gpu_memory_usage("After building vllm rollout")

        self.rollout_sharding_manager = FSDPVLLMShardingManager(
            module=self.fsdp_module,
            inference_engine=self.rollout.inference_engine,
            device_mesh=rollout_device_mesh,
        )
        print_gpu_memory_usage("After building sharding manager")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if self._is_critic:
            model_config = self.config.critic.model
            fsdp_config = self.config.critic.fsdp
            optim_config = self.config.critic.optim
            padding_free = self.config.critic.padding_free
        elif self._is_actor:
            model_config = self.config.actor.model
            fsdp_config = self.config.actor.fsdp
            optim_config = self.config.actor.optim
            padding_free = self.config.actor.padding_free
        elif self._is_ref:
            model_config = self.config.actor.model
            fsdp_config = self.config.ref.fsdp
            optim_config = None
            padding_free = self.config.ref.padding_free
        else:
            raise ValueError("Unknown role.")

        if self._is_actor or self._is_critic or self._is_ref:
            self._build_model_optimizer(
                model_config=model_config,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                padding_free=padding_free,
            )
            # get the original unwrapped module
            self.unwrapped_model = self.fsdp_module._fsdp_wrapped_module
            if self._use_optimizer_offload:
                offload_fsdp_optimizer(optimizer=self.optimizer)
                print_gpu_memory_usage("After offload actor optimizer during init")

        if self._is_actor:
            self.actor = DataParallelPPOActor(
                config=self.config.actor,
                actor_module=self.fsdp_module,
                actor_optimizer=self.optimizer,
            )

        if self._is_critic:
            self.critic = DataParallelPPOCritic(
                config=self.config,
                critic_module=self.fsdp_module,
                critic_optimizer=self.optimizer,
            )

        if self._is_rollout:
            self._build_rollout()

        if self._is_ref:
            self.ref_policy = DataParallelPPOActor(
                config=self.config.ref,
                actor_module=self.fsdp_module,
            )

        if self._is_actor or self._is_critic:
            self.flops_counter = FlopsCounter(self.model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.fsdp_module,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
            )

        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, path: str, global_step: int = 0, remove_previous_ckpt: bool = False):
        assert self._is_actor or self._is_critic
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        self.checkpoint_manager.save_checkpoint(
            local_path=path,
            global_step=global_step,
            remove_previous_ckpt=remove_previous_ckpt,
        )
        dist.barrier()
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path: str, remove_ckpt_after_load: bool = False):
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        self.checkpoint_manager.load_checkpoint(path=path, remove_ckpt_after_load=remove_ckpt_after_load)
        dist.barrier()
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

    """ActorRolloutRefWorker"""

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        assert self._is_actor
        data = data.to("cuda")

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            load_fsdp_optimizer(optimizer=self.optimizer)

        print_gpu_memory_usage("Before update policy")
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)

            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["mfu/actor"] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size

            self.lr_scheduler.step()
            lr = self.lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr
            print_gpu_memory_usage("After update policy")

            # TODO: here, we should return all metrics
            output = DataProto(meta_info={"metrics": metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            offload_fsdp_optimizer(optimizer=self.optimizer)

        output = output.to("cpu")
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        assert self._is_rollout
        prompts = prompts.to("cuda")

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        with self.rollout_sharding_manager:
            # after parameters sync with rollout, offload actor model to CPU
            if self._use_param_offload:
                offload_fsdp_model(self.fsdp_module)

            if self._use_optimizer_offload:
                offload_fsdp_optimizer(optimizer=self.optimizer)

            print_gpu_memory_usage("After entering rollout sharding manager")
            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            output = self.rollout.generate_sequences(prompts=prompts)
            output = self.rollout_sharding_manager.postprocess_data(output)
            print_gpu_memory_usage("After rollout generation")

        output = output.to("cpu")
        torch.cuda.empty_cache()  # clear kv cache
        print_gpu_memory_usage("After recompute log prob")
        return output

    def _process_multi_modal_inputs(self, data: DataProto):
        responses = data.non_tensor_batch["response"]
        raw_video_inputs = data.non_tensor_batch["raw_video_inputs"]
        batch_multi_modal_inputs = []
        batch_input_ids = []
        batch_attention_masks = []
        batch_position_ids = []
        batch_raw_prompt_ids = []
        batch_multi_modal_data = []
        for example_id, raw_video_input in enumerate(raw_video_inputs):
            video_inputs = torch.tensor(raw_video_input["video"])[0]
            raw_fps_inputs = raw_video_input["fps"]
            raw_question = raw_video_input["raw_question"]
            raw_question = self.tokenizer.decode(raw_question, skip_special_tokens=True)
            responses_per_example = responses[example_id * self.config.rollout.n: (example_id + 1) * self.config.rollout.n]
            responses_per_example = self.tokenizer.batch_decode(responses_per_example, skip_special_tokens=True)
            for response in responses_per_example:
                pred_glues = []
                pred_glue = re.search(r'<glue>(.*?)</glue>', response, re.DOTALL)
                if pred_glue:
                    pred_glue = pred_glue.group(1).strip()
                    if is_valid_two_d_list_format(pred_glue):
                        pred_glues = ast.literal_eval(pred_glue)
                frame_indices = []
                if pred_glues:
                    for pred_glue in pred_glues:
                        start_time, end_time = pred_glue
                        start_frame = max(0, int(start_time * raw_fps_inputs))
                        end_frame = min(int(end_time * raw_fps_inputs), video_inputs.shape[0] - 1)
                        frame_indices.extend(list(range(start_frame, end_frame)))   
                if len(frame_indices) > 0:
                    frame_indices = sorted(list(set(frame_indices)))
                    if len(frame_indices) > 256:
                        select_indices = torch.linspace(0, len(frame_indices) - 1, 256).round().long().tolist()
                        frame_indices = [frame_indices[i] for i in select_indices]
                    crop_frames = video_inputs[frame_indices]
                else:
                    if video_inputs.shape[0] > 20:
                        frame_indices = list(range(0, video_inputs.shape[0], video_inputs.shape[0] // 20))
                    else:
                        frame_indices = list(range(0, video_inputs.shape[0]))
                    crop_frames = video_inputs[frame_indices]
                
                # print("pred_glues", pred_glues, video_inputs.shape[0], raw_fps_inputs, len(frame_indices))
                crop_frames, sample_fps = resize_video(crop_frames, raw_fps_inputs)

                input_text = raw_question
                if self.config.reward.reason_reward:
                    think_content_match = re.search(r'<think>(.*)</think>', response, re.DOTALL)
                    if think_content_match:
                        think_content = think_content_match.group(1)
                        cleaned_text = re.sub(r'<time>.*?</time>', '', think_content)
                        cleaned_text = cleaned_text.strip()
                        input_text = raw_question + " <think>" + cleaned_text + "</think>"

                messages = [
                    {"role": "user", "content": [
                        {
                            "type": "video", 
                            "total_pixels": 4096*28*28, 
                            "min_pixels": 16*28*28,
                            "video": "test.mp4",
                            "fps": 1.0,
                        },
                        {
                            "type": "text", 
                            "text": input_text
                        },
                    ]},
                ]
                prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                model_inputs = self.processor(
                    videos=[crop_frames], text=[prompt], add_special_tokens=False, return_tensors="pt"
                )
                model_inputs.pop("second_per_grid_ts")
                input_ids = model_inputs.pop("input_ids")[0]
                attention_mask = model_inputs.pop("attention_mask")[0]
                batch_multi_modal_inputs.append(dict(model_inputs))
                batch_multi_modal_data.append({"video": crop_frames})

                position_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids,
                    image_grid_thw=model_inputs.get("image_grid_thw", None),
                    video_grid_thw=model_inputs.get("video_grid_thw", None),
                    second_per_grid_ts=[2.0 / sample_fps],
                    attention_mask=attention_mask,
                )

                input_ids, attention_mask, position_ids = postprocess_data(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    max_length=self.config.rollout.max_prompt_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    left_pad=True,
                    truncation="right",
                )

                raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                if len(raw_prompt_ids) > self.config.rollout.max_prompt_length:
                    raw_prompt_ids = raw_prompt_ids[: self.config.rollout.max_prompt_length]
                
                batch_input_ids.append(input_ids)
                batch_attention_masks.append(attention_mask)
                batch_position_ids.append(position_ids)
                batch_raw_prompt_ids.append(raw_prompt_ids)
        
        batch_input_ids = torch.stack(batch_input_ids)
        batch_attention_masks = torch.stack(batch_attention_masks)
        batch_position_ids = torch.stack(batch_position_ids)
        batch_raw_prompt_ids = np.array(batch_raw_prompt_ids, dtype=object)
        batch_multi_modal_inputs = np.array(batch_multi_modal_inputs, dtype=object)
        batch_multi_modal_data = np.array(batch_multi_modal_data, dtype=object)

        batch = TensorDict(
            {
                "input_ids": batch_input_ids,
                "attention_mask": batch_attention_masks,
                "position_ids": batch_position_ids,
            },
            batch_size=batch_input_ids.shape[0],
        )
        non_tensor_batch = TensorDict(
            {
                "raw_prompt_ids": batch_raw_prompt_ids,
                "multi_modal_inputs": batch_multi_modal_inputs,
                "multi_modal_data": batch_multi_modal_data,
            }
        )
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences_zoom(self, prompts: DataProto):
        prompts = self._process_multi_modal_inputs(prompts)
        prompts = prompts.to("cuda")

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        with self.ulysses_sharding_manager:
            prompts = self.ulysses_sharding_manager.preprocess_data(prompts)
            output = self.ref_policy.generate_sequences_zoom(data=prompts)
            output = self.ulysses_sharding_manager.postprocess_data(output)
            batch = TensorDict(
                {
                    "responses_zoom": output.to("cpu")
                },
                batch_size=prompts.batch["input_ids"].shape[0],
            )

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.fsdp_module._handle.reshard(True)

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        torch.cuda.empty_cache()
        return DataProto(batch=batch)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto):
        assert self._is_actor
        data = data.to("cuda")
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["temperature"] = self.config.rollout.temperature
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.actor.compute_log_prob(data=data)
            output = DataProto.from_dict(
                tensors={"old_log_probs": output}, meta_info={"temperature": self.config.rollout.temperature}
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.fsdp_module._handle.reshard(True)

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        output = output.to("cpu")
        torch.cuda.empty_cache()
        print_gpu_memory_usage("After compute_log_prob")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        assert self._is_ref
        data = data.to("cuda")
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        data.meta_info["temperature"] = self.config.rollout.temperature
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.ref_policy.compute_log_prob(data=data)
            output = DataProto.from_dict(tensors={"ref_log_prob": output})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.fsdp_module._handle.reshard(True)

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        output = output.to("cpu")
        torch.cuda.empty_cache()
        print_gpu_memory_usage("After compute_ref_log_prob")
        return output

    """CriticWorker"""

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        assert self._is_critic
        data = data.to("cuda")
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={"values": values})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        output = output.to("cpu")
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        data = data.to("cuda")
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            load_fsdp_optimizer(optimizer=self.optimizer)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name="update_critic", logger=None) as timer:
                metrics = self.critic.update_critic(data=data)

            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["mfu/critic"] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size

            self.lr_scheduler.step()
            lr = self.lr_scheduler.get_last_lr()[0]
            metrics["critic/lr"] = lr

            output = DataProto(batch=None, meta_info={"metrics": metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            offload_fsdp_optimizer(optimizer=self.optimizer)

        output = output.to("cpu")
        torch.cuda.empty_cache()
        return output
