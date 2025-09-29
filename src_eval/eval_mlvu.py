# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# pyre-strict

# Need to call this before importing transformers.


import datetime
import json
import logging
import os
import re
import shutil
import uuid
from itertools import chain
import argparse
import random
import sys
import numpy as np
import pysubs2
from src_eval.my_qwen_utils import process_vision_info

import torch

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

from decord import cpu, VideoReader
from torch import distributed as dist
from tqdm import tqdm
import ast

from transformers.trainer_pt_utils import IterableDatasetShard

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

QUESTION_TEMPLATE_QA = """Answer the question: "[QUESTION]" according to the content of the video. Select the answer from: [OPTION]. Reply only with the letter of the option.
"""

def is_valid_two_d_list_format(s):
    pattern = r'^\[(\(\d+(\.\d+)?,\s*\d+(\.\d+)?\)(,\s*\(\d+(\.\d+)?,\s*\d+(\.\d+)?\))*(,)?|)\]$'
    if not re.match(pattern, s):
        return False
    try:
        lst = ast.literal_eval(s)
        if not isinstance(lst, list):
            return False
        for item in lst:
            if not isinstance(item, tuple):
                return False
            if len(item) != 2:
                return False
            for num in item:
                if not isinstance(num, (int, float)):
                    return False
            if item[0] > item[1]:
                return False
        return True
    except:
        return False

class EvalDataset(torch.utils.data.IterableDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
    ) -> None:
        super(EvalDataset, self).__init__()

        self.data_path = data_path

        self.data_list = {
            "count": ("json/4_count.json", f"video/count", "video"),
            "ego": ("json/3_ego.json", f"video/ego", "video"),
            "needle": ("json/2_needle.json", f"video/needle", "video"),
            "order": ("json/5_order.json", f"video/order", "video"),
            "plotQA": ("json/1_plotQA.json", f"video/plotQA", "video"),
            "anomaly_reco": (
                "json/6_anomaly_reco.json",
                f"video/anomaly_reco",
                "video",
            ),
            "topic_reasoning": (
                "json/7_topic_reasoning.json",
                f"video/topic_reasoning",
                "video",
            ),
        }

        list_data_dict = []
        for k, v in self.data_list.items():
            with open(os.path.join(data_path, v[0]), "r") as f:
                json_data = json.load(f)
            for item in json_data:
                answer = item["answer"]
                answer_idx = 0
                for idx, c in enumerate(item["candidates"]):
                    if c == answer:
                        answer_idx = idx
                        break
                list_data_dict.append(
                    {
                        "task_type": k,
                        "video": os.path.join(self.data_path, v[1], item["video"]),
                        "video_name": item["video"].split(".")[0],
                        "question": item["question"],
                        "answer": chr(ord('A') + answer_idx),
                        "choices": [f"({chr(ord('A') + idx)}) {c}" for idx, c in enumerate(item["candidates"])]
                    }
                )

        # pyre-fixme[4]: Attribute must be annotated.
        self.data = list_data_dict
        random.seed(42)
        random.shuffle(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, i):
        return self.data[i]


def train(args) -> None:
    # Initialize distributed training
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    torch.distributed.barrier()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # Load model and processor
    logging.info(f"Rank {local_rank}: Loading model from {args.model_path}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    model.to("cuda")
    processor = AutoProcessor.from_pretrained(args.model_path)
    model_name = args.model_path.split("/")[-1]

    # Prepare dataset
    dataset = EvalDataset(
        data_path=args.data_path,
    )
    world_size = torch.distributed.get_world_size()
    world_rank = torch.distributed.get_rank()

    # Define checkpoint file path
    checkpoint_dir = f"work_dirs_eval/{model_name}/MLVU/{args.prefix}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, f"cuda:{world_rank}.json")

    output = []
    # Use a set for quick lookup of processed item identifiers (video_name, question_text)
    processed_identifiers = set() 

    # Load checkpoint if it exists
    if os.path.exists(checkpoint_file):
        logging.info(f"Rank {world_rank}: Loading checkpoint from {checkpoint_file}...")
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                output = checkpoint_data.get('output', [])
                # Reconstruct processed_identifiers from loaded output
                # Each item in 'output' should contain 'videoID' and 'question'
                for item in output:
                    if 'videoID' in item and 'question' in item:
                        identifier = (item['videoID'], item['question']) 
                        processed_identifiers.add(identifier)
                    else:
                        logging.warning(f"Rank {world_rank}: Skipping malformed item in checkpoint output: {item}")
            logging.info(f"Rank {world_rank}: Resuming with {len(output)} already processed question-video pairs.")
        except json.JSONDecodeError:
            logging.error(f"Rank {world_rank}: Error decoding checkpoint file {checkpoint_file}. Starting fresh.")
            output = []
            processed_identifiers = set()
    else:
        logging.info(f"Rank {world_rank}: No checkpoint found. Starting fresh.")

    shard_dataset = IterableDatasetShard(
        dataset,
        batch_size=1, 
        num_processes=world_size,
        process_index=world_rank,
    )
    torch.distributed.barrier()
    
    total_videos_for_rank = len(list(IterableDatasetShard(
        dataset,
        batch_size=1,
        num_processes=world_size,
        process_index=world_rank,
    )))

    pbar = tqdm(shard_dataset, total=total_videos_for_rank, desc=f"Rank {world_rank} Processing Videos")

    for line in pbar:
        video_name = line["video_name"]
        question = line["question"]
        task_type = line["task_type"]
        video_path = line["video"]
        choices = line["choices"]
        answer = line["answer"]
        
        current_identifier = (video_name, question)

        if current_identifier in processed_identifiers:
            pbar.set_postfix_str(f"Skipped: {video_name[:8]}... - {question[:20]}...")
            continue # Skip to the next question for this video

        options = " ".join(choices)
        input_text = f"""{QUESTION_TEMPLATE_QA.replace("[QUESTION]", question).replace("[OPTION]", options)}"""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 460800,
                        "total_pixels": args.total_pixels * 28 * 28,
                        "timestamps": [],
                        "fps": args.fps,
                    },
                    {"type": "text", "text": input_text},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True, maximum_frames=args.nframes)
        inputs = processor(
            text=[text],
            images=None,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        output.append(
            {
            'videoID': video_name, 
            'question': question,         
            'answer': answer,
            'options': choices,
            'output': output_text[0].replace("<answer>", "").replace("</answer>", ""),
            'task_type': task_type,
            }
        )

        processed_identifiers.add(current_identifier)

        print(f"Rank {world_rank} Output for {video_name[:8]}... - {question[:20]}...: {output_text[0]}, answer: {answer}", flush=True)

        with open(checkpoint_file, 'w') as f:
            json.dump({'output': output, 'processed_identifiers': list(processed_identifiers)}, f)
        
        pbar.set_postfix_str(f"Processed: {video_name[:8]}... - {question[:20]}...")

    dist.barrier()
    
    final_output = [None] * world_size
    dist.all_gather_object(
        final_output,
        output,
    )
    all_output = list(chain(*final_output)) 

    global_rank = dist.get_rank()
    if global_rank == 0:
        output_filename = os.path.join(checkpoint_dir, f"output.json")
        logging.info(f"Rank 0: All processes completed. Saving final results to {output_filename}")
        with open(output_filename, "w") as f:
            json.dump(all_output, f) 
        
        result = {}
        for task_type in dataset.data_list.keys():
            task_type_output = [item for item in all_output if item['task_type'] == task_type]
            accuracy = sum(1 for item in task_type_output if item['answer'] in item['output'] or item['output'] in item['answer']) / len(task_type_output)
            result[task_type] = accuracy
            logging.info(f"Rank 0: Accuracy for {task_type}: {accuracy}")
        result["overall"] = sum(1 for item in all_output if item['answer'] in item['output'] or item['output'] in item['answer']) / len(all_output)
        
        result_filename = os.path.join(checkpoint_dir, f"result.json")
        with open(result_filename, "w") as f:
            json.dump(result, f)

        logging.info("Rank 0: Initiating checkpoint file cleanup.")
        for rank_idx in range(world_size):
            rank_checkpoint_file = os.path.join(checkpoint_dir, f"cuda:{rank_idx}.json")
            if os.path.exists(rank_checkpoint_file):
                os.remove(rank_checkpoint_file)
                logging.info(f"Rank 0: Removed checkpoint file for rank {rank_idx}.")
            else:
                logging.info(f"Rank 0: Checkpoint file for rank {rank_idx} not found (already removed or never created).")
        logging.info("Rank 0: All processing complete and checkpoints removed.")
    
    dist.barrier() 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate video-MME dataset with checkpointing.")

    parser.add_argument('--model_path', default="", required=True)
    parser.add_argument('--data_path', default="./dataset/MLVU/MLVU", required=True)
    parser.add_argument('--nframes', default=256, type=int,
                        help="Number of frames to extract from each video.")
    parser.add_argument('--fps', default=2.0, type=float,
                        help="FPS of the video.")
    parser.add_argument('--total_pixels', default=16384, type=int, help="Total pixels of the video.")
    parser.add_argument("--num_gpus", type=int, default=8,
                        help="Number of GPUs to use (informational, usually set by environment).")
    parser.add_argument('--prefix', default="none", type=str)
    args = parser.parse_args()
    train(args)