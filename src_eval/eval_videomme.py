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

import torch

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from src_eval.my_qwen_utils import process_vision_info, read_subtitle

from decord import cpu, VideoReader
from pyarrow import parquet as pq
from torch import distributed as dist
from tqdm import tqdm
import ast

from transformers.trainer_pt_utils import IterableDatasetShard

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

QUESTION_TEMPLATE_QA = """Answer the question: "[QUESTION]" according to the content of the video. Select the answer from: [OPTION]. Reply only with the letter of the option.
"""

# QUESTION_TEMPLATE_QA = """
# Answer the question: "[QUESTION]" according to the content of the video. Select the answer from: [OPTION]. Reply only with the letter of the option.\n
# Provide your answer within the <answer> </answer> tags, output the corresponding letter of the option. At the same time, in the <glue> </glue> tags, include only the precise video segments (in seconds) that strongly support your answer, in the format of [(s1, e1), (s2, e2), ...]. For example: <answer>A</answer>\n<glue>[(20.3, 30.8)]</glue>.
# """

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

def load_parquet(parquet_file):
    table = pq.read_table(parquet_file)
    df = table.to_pandas()

    jsons = []
    video_id_map = {} # To store video_id to list index mapping for efficient lookup

    for record in df.itertuples():
        video_id_int = int(record.video_id)
        
        if video_id_int not in video_id_map:
            # If video_id not seen, create a new entry and map its ID to current list length
            new_entry = {
                "video_id": record.video_id,
                "youtube_id": record.videoID,
                "url": record.url,
                "duration": record.duration,
                "domain": record.domain,
                "sub_category": record.sub_category,
                "questions": [] # Initialize questions list
            }
            jsons.append(new_entry)
            video_id_map[video_id_int] = len(jsons) - 1 # Store the index
        
        # Append the question to the correct video entry using the map
        jsons[video_id_map[video_id_int]]["questions"].append(
            {
                "question_id": record.question_id,
                "task_type": record.task_type,
                "question": record.question,
                "choices": list(record.options),
                "answer": record.answer,
            }
        )
    return jsons


class EvalDataset(torch.utils.data.IterableDataset):
    """Dataset for supervised fine-tuning."""

    video_formats = [".mp4", ".avi", ".mov", ".mkv"]

    def __init__(
        self,
        data_path: str,
    ) -> None:
        super(EvalDataset, self).__init__()
        logging.info("Loading data...")

        self.data_path = data_path

        data_list = load_parquet(
            os.path.join(self.data_path, "videomme/test-00000-of-00001.parquet")
        )

        list_data_dict = []

        for item in data_list:
            video_ytid = item["url"].split("watch?v=")[-1]
            video_path = os.path.join(self.data_path, "data", f"{video_ytid}.mp4")
            found_video = False
            for fmt in self.video_formats:
                temp_path = os.path.join(self.data_path, "data", f"{video_ytid}{fmt}")
                if os.path.exists(temp_path):
                    video_path = temp_path
                    found_video = True
                    break
            if not found_video:
                logging.warning(f"Video file not found for {video_ytid}. Skipping this entry.")
                continue

            subtitle_path = os.path.join(
                self.data_path, "subtitle", f"{video_ytid}.srt"
            )

            list_data_dict.append(
                {
                    "questions": item["questions"],
                    "video": video_path,
                    "subtitle": subtitle_path, # Subtitle path is included, though not directly used in the current prompt format
                    "video_name": video_ytid, # Used as part of the unique identifier
                    "duration": item["duration"],
                }
            )

        self.data = list_data_dict
        random.seed(42)
        random.shuffle(self.data) # This shuffle is handled by tracking unique (video_name, question) pairs

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
    checkpoint_dir = f"work_dirs_eval/{model_name}/VideoMME/{args.prefix}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, f"cuda:{world_rank}.json")

    output = []
    processed_identifiers = set() 

    if os.path.exists(checkpoint_file):
        logging.info(f"Rank {world_rank}: Loading checkpoint from {checkpoint_file}...")
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                output = checkpoint_data.get('output', [])
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
        batch_size=1, # Process one video at a time
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
        questions = line["questions"]
        duration_type = line["duration"]
        video_path = line["video"]
        subtitle_path = line["subtitle"]
        subtitle_time_pair = None
        if "sub" in args.prefix and os.path.exists(subtitle_path):
            subtitle_text = read_subtitle(subtitle_path)
            subtitle_time_pair = list(subtitle_text.items())

        for question in questions:
            q = question["question"]
            
            current_identifier = (video_name, q)

            if current_identifier in processed_identifiers:
                pbar.set_postfix_str(f"Skipped: {video_name[:8]}... - {q[:20]}...")
                continue # Skip to the next question for this video

            if duration_type not in args.duration:
                processed_identifiers.add(current_identifier)
                with open(checkpoint_file, 'w') as f:
                    json.dump({'output': output, 'processed_identifiers': list(processed_identifiers)}, f)
                pbar.set_postfix_str(f"Skipped duration: {video_name[:8]}... - {q[:20]}...")
                continue 

            options = " ".join(question["choices"])
            
            input_text = f"""{QUESTION_TEMPLATE_QA.replace("[QUESTION]", q).replace("[OPTION]", options)}"""
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                            "max_pixels": 460800,
                            "nframes": args.nframes,
                            "total_pixels": args.total_pixels * 28 * 28,
                            "extend": False,
                        },
                        {"type": "text", "text": input_text},
                    ],
                }
            ]
            
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            
            if subtitle_time_pair:
                subtitle_text = "\n".join([f"{text}" for time, text in subtitle_time_pair])
                text = text.replace("<|vision_start|><|video_pad|><|vision_end|>", subtitle_text + "<|vision_start|><|video_pad|><|vision_end|>")
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
                'question': q,         
                'answer': question["answer"],
                'options': question["choices"],
                'output': output_text[0].replace("<answer>", "").replace("</answer>", ""),
                'duration': duration_type,
                }
            )

            processed_identifiers.add(current_identifier)

            # Print current output to console for real-time monitoring
            print(f"Rank {world_rank} Output for {video_name[:8]}... - {q[:20]}...: {output_text[0]}, answer: {question['answer']}", flush=True)

            with open(checkpoint_file, 'w') as f:
                json.dump({'output': output, 'processed_identifiers': list(processed_identifiers)}, f)
            
            pbar.set_postfix_str(f"Processed: {video_name[:8]}... - {q[:20]}...")


    dist.barrier()
    
    final_output = [None] * world_size
    dist.all_gather_object(
        final_output,
        output,
    )
    all_output = list(chain(*final_output)) # Flatten the list of lists from all ranks

    global_rank = dist.get_rank()
    if global_rank == 0:
        output_filename = os.path.join(checkpoint_dir, f"output.json")
        logging.info(f"Rank 0: All processes completed. Saving final results to {output_filename}")
        with open(output_filename, "w") as f:
            json.dump(all_output, f) 
        
        result = {}
        for duration in args.duration:
            duration_output = [item for item in all_output if item['duration'] == duration]
            accuracy = sum(1 for item in duration_output if item['answer'] in item['output'] or item['output'] in item['answer']) / len(duration_output)
            result[duration] = accuracy
            logging.info(f"Rank 0: Accuracy for {duration}: {accuracy}")
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
    parser.add_argument('--data_path', default="./dataset/Video-MME", required=True)
    parser.add_argument('--duration', default="long,medium,short", type=str,
                        help="Comma-separated list of video durations to process (e.g., 'long,medium,short').")
    parser.add_argument('--nframes', default=512, type=int,
                        help="Number of frames to extract from each video.")
    parser.add_argument("--num_gpus", type=int, default=8,
                        help="Number of GPUs to use (informational, usually set by environment).")
    parser.add_argument('--total_pixels', default=16384, type=int, help="Total pixels of the video.")
    parser.add_argument('--prefix', default="none", type=str)
    parser.add_argument('--fps', default=2.0, type=float, help="FPS of the video.")
    args = parser.parse_args()
    args.duration = args.duration.split(",") 

    train(args)