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
from src_eval.my_qwen_utils import process_vision_info, confidence_score, fetch_video, resize_video

import torch

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

from decord import cpu, VideoReader
from torch import distributed as dist
from tqdm import tqdm
import ast

from transformers.trainer_pt_utils import IterableDatasetShard

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

QUESTION_TEMPLATE_QA = """Answer the question: "[QUESTION]" according to the content of the video. Select the answer from: [OPTION]. Reply only with the letter of the option within <answer></answer>.
"""

QUESTION_TEMPLATE_QA_REASON = """Answer the question: "[QUESTION]" according to the content of the video. Select the answer from: [OPTION]. Reply only with the letter of the option.\n
Provide your answer within the <answer> </answer> tags, output the corresponding letter of the option. At the same time, in the <glue> </glue> tags, include only the precise video segments (in seconds) that strongly support your answer, in the format of [(s1, e1), (s2, e2), ...]. For example: <answer>A</answer>\n<glue>[(20.3, 30.8)]</glue>.
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
            # If checkpoint file is corrupted, start fresh
            output = []
            processed_identifiers = set()
    else:
        logging.info(f"Rank {world_rank}: No checkpoint found. Starting fresh.")

    # Create shard for this process
    shard_dataset = IterableDatasetShard(
        dataset,
        batch_size=1, # Process one video at a time
        num_processes=world_size,
        process_index=world_rank,
    )
    torch.distributed.barrier()
    
    # Calculate total items for tqdm. Note: This will iterate the dataset once.
    # For very large datasets, you might want to pre-calculate this or handle tqdm differently.
    total_videos_for_rank = len(list(IterableDatasetShard(
        dataset,
        batch_size=1,
        num_processes=world_size,
        process_index=world_rank,
    )))

    pbar = tqdm(shard_dataset, total=total_videos_for_rank, desc=f"Rank {world_rank} Processing Videos")

    def model_inference(video_input, fps_inputs, input_prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": "test.mp4",
                        "timestamps": [],
                        "extend": False,
                        "fps": args.fps,
                    },
                    {"type": "text", "text": input_prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text],
            images=None,
            videos=[video_input],
            padding=True,
            return_tensors="pt",
            fps=[fps_inputs],
        )
        inputs = inputs.to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=512, return_dict_in_generate=True, output_logits=True)
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs.sequences[0][input_length:]
        output_text = processor.decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        answer_confidence, glue_confidence = confidence_score(torch.stack(outputs.logits)[:, 0], generated_tokens, processor)
        return output_text, answer_confidence, glue_confidence

    for line in pbar:
        video_name = line["video_name"]
        question = line["question"]
        task_type = line["task_type"]
        video_path = line["video"]
        choices = line["choices"]
        answer = line["answer"]
        
        # Create a unique identifier for this specific question-video pair
        current_identifier = (video_name, question)

        # Skip if this item has already been processed in a previous run or earlier in this run
        if current_identifier in processed_identifiers:
            pbar.set_postfix_str(f"Skipped: {video_name[:8]}... - {question[:20]}...")
            continue # Skip to the next question for this video
        # --- Actual Model Inference ---
        options = " ".join(choices)
        
        video_inputs_raw, raw_fps_inputs = fetch_video({"video": video_path, "fps": args.fps}, return_video_sample_fps=True, resize=False)
        whole_video_duration = video_inputs_raw.shape[0] / raw_fps_inputs
        start_seconds = 0
        answers = []
        pred_glues = []
        answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
        pred_pattern = re.compile(r'<glue>(.*?)</glue>', re.DOTALL)
        if video_inputs_raw.shape[0] <= args.chunk_size or task_type == "order" or task_type == "count":
            video_inputs_split = [video_inputs_raw]
            input_prompt = f"""{QUESTION_TEMPLATE_QA.replace("[QUESTION]", question).replace("[OPTION]", options)}"""
        else:
            video_inputs_split = video_inputs_raw.split(args.chunk_size)
            input_prompt = f"""{QUESTION_TEMPLATE_QA_REASON.replace("[QUESTION]", question).replace("[OPTION]", options)}"""
        for video_input in video_inputs_split:
            video_input, fps_inputs = resize_video(video_input, raw_fps_inputs, total_pixels=args.total_pixels * 28 * 28)
            output_text, answer_confidence, glue_confidence = model_inference(video_input, fps_inputs, input_prompt)
            end_seconds = start_seconds + video_input.shape[0] / fps_inputs
            match_answer = answer_pattern.search(output_text)
            if match_answer:
                answer = match_answer.group(1).strip().replace('(', '').replace(')', '')
            else:
                answer = "C"
            answers.append({"answer": answer, "confidence": round(answer_confidence, 4)})
            
            match_pred = pred_pattern.search(output_text)
            if match_pred and is_valid_two_d_list_format(match_pred.group(1).strip()):
                pred_glue = ast.literal_eval(match_pred.group(1).strip())
            else:
                pred_glue = []
            shift_pred_glue = []
            for glue in pred_glue:
                if len(glue)  == 2:
                    start_second = round(glue[0] + start_seconds, 2)
                    end_second = round(min(glue[1] + start_seconds, end_seconds), 2)
                    shift_pred_glue.append((start_second, end_second))
            pred_glues.append({"pred": shift_pred_glue, "confidence": round(glue_confidence, 2)})
            start_seconds = end_seconds

        top_answers, top_indices = zip(*sorted(zip(answers, range(len(answers))), key=lambda x: x[0]['confidence'], reverse=True)[:args.top_n])
        
        if len(video_inputs_split) > 0:
            input_prompt = f"""{QUESTION_TEMPLATE_QA.replace("[QUESTION]", question).replace("[OPTION]", options)}"""
            video_inputs, fps_inputs = resize_video(video_inputs_raw, raw_fps_inputs, total_pixels=args.total_pixels * 28 * 28)
            output_text, coarse_answer_confidence, glue_confidence = model_inference(video_inputs, fps_inputs, input_prompt)
            match_answer = answer_pattern.search(output_text)
            if match_answer:
                coarse_answer = match_answer.group(1).strip().replace('(', '').replace(')', '')
            else:
                coarse_answer = "C"
        else:
            coarse_answer = top_answers[0]['answer']
            coarse_answer_confidence = round(top_answers[0]['confidence'], 2)
        
        
        if len(answers) == 1 or (top_answers[0]['confidence'] == 1.0 and len(set([answer['answer'] for answer in top_answers if answer['confidence'] == 1.0])) == 1):
            fine_answer = answers[0]['answer']
            fine_answer_confidence = round(top_answers[0]['confidence'], 2)
        else:
            timestamps = []
            pred_timestamps = [pred_glues[i] for i in list(top_indices)]
            for answer, glue in zip(top_answers, pred_timestamps):
                if glue['confidence'] > 10 and len(glue['pred']) > 0:
                    timestamps.extend(glue['pred'])
            video_inputs_raw, raw_fps_inputs = fetch_video({"video": video_path, "fps": args.fps, "timestamps": timestamps}, return_video_sample_fps=True, resize=False)
            video_inputs, fps_inputs = resize_video(video_inputs_raw, raw_fps_inputs, total_pixels=args.total_pixels * 28 * 28)
            input_prompt = f"""{QUESTION_TEMPLATE_QA.replace("[QUESTION]", question).replace("[OPTION]", options)}"""
            output_text, fine_answer_confidence, glue_confidence = model_inference(video_inputs, fps_inputs, input_prompt)
            match_answer = answer_pattern.search(output_text)
            if match_answer:
                fine_answer = match_answer.group(1).strip().replace('(', '').replace(')', '')
            else:
                fine_answer = top_answers[0]['answer']

        final_answer, final_confidence = (fine_answer, fine_answer_confidence) if fine_answer_confidence > coarse_answer_confidence else (coarse_answer, coarse_answer_confidence)
        if final_confidence < 0.6 and final_confidence < top_answers[0]['confidence']:
            final_answer, final_confidence = top_answers[0]['answer'], top_answers[0]['confidence']
        
        output.append(
            {
            'videoID': video_name, # The unique video identifier
            'question': question,         # The unique question text
            'answer': line["answer"],
            'options': choices,
            'task_type': task_type,
            'duration': whole_video_duration,
            'pred': {"pred_glues": pred_glues, "answers": answers},
            "fine_answer": (fine_answer, round(fine_answer_confidence, 2)),
            "coarse_answer": (coarse_answer, round(coarse_answer_confidence, 2)),
            "pred_answer": final_answer
            }
        )

        print(output[-1], flush=True)

        # Mark this specific question-video pair as processed
        processed_identifiers.add(current_identifier)

        # Print current output to console for real-time monitoring
        print(f"Rank {world_rank} Output pred_answer: {answers}, final_answer: {final_answer}, answer: {line['answer']}", flush=True)

        # Save checkpoint after each question for maximum resilience.
        # For very large datasets, consider saving every N questions/videos to reduce I/O overhead.
        with open(checkpoint_file, 'w') as f:
            json.dump({'output': output, 'processed_identifiers': list(processed_identifiers)}, f)


    # --- End of Processing Loop ---

    # Ensure all processes have finished their local computations before gathering results
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
            json.dump(all_output, f) # Added indent for readability
        
        result = {}
        for task_type in dataset.data_list.keys():
            task_type_output = [item for item in all_output if item['task_type'] == task_type]
            accuracy = sum(1 for item in task_type_output if item['answer'] in item['pred_answer'] or item['pred_answer'] in item['answer']) / len(task_type_output)
            result[task_type] = accuracy
            logging.info(f"Rank 0: Accuracy for {task_type}: {accuracy}")
        result["overall"] = sum(1 for item in all_output if item['answer'] in item['pred_answer'] or item['pred_answer'] in item['answer']) / len(all_output)
        
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
    
    dist.barrier() # Final barrier to ensure all ranks acknowledge cleanup before exiting

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate video-MME dataset with checkpointing.")
    parser.add_argument('--model_path', default="", required=True)
    parser.add_argument('--data_path', default="./dataset/MLVU/MLVU", required=True)
    parser.add_argument('--fps', default=1.0, type=float)
    parser.add_argument('--total_pixels', default=16384, type=int)
    parser.add_argument('--prefix', default="none", type=str)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--top_n", type=int, default=5)
    args = parser.parse_args()
    train(args)