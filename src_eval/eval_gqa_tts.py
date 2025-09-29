from torch import distributed as dist
import argparse
import numpy as np
import json
from tqdm import tqdm
import os
import re
import pickle
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from src_eval.my_qwen_utils import process_vision_info, confidence_score
import random
import ast
import os
import json
from math import ceil

QUESTION_TEMPLATE = """
Output key information relevant to the question and options, marking precise timestamps or time ranges in seconds within <time> </time> tags, and present them in an interleaved analysis format. Enclose the full analysis in <think> </think> tags. For example: <think> After folding the face towel <time>(5.2, 10.4)</time>, the person placed it on the bed <time>(20.3, 30.8)</time>.</think>\n
Then, provide your answer within the <answer> </answer> tags, output the corresponding letter of the option. At the same time, in the <glue> </glue> tags, include only the precise video segments (in seconds) that strongly support your answer, in the format of [(s1, e1), (s2, e2), ...]. Do not list unrelated time ranges. For example: <answer>A</answer>\n<glue>[(20.3, 30.8)]</glue>.
"""

def split_data(data, num_gpus):
    is_dict = isinstance(data, dict)

    if is_dict:
        data = list(data.items())
    elif not isinstance(data, list):
        data = list(data)

    data_size = len(data)
    chunk_size = ceil(data_size / num_gpus)  
    chunks = [data[i * chunk_size:(i + 1) * chunk_size] for i in range(num_gpus)]

    if is_dict:
        chunks = [dict(chunk) for chunk in chunks]

    return chunks

VIDEO_INFO_CACHE = {}

def get_args():
    parser = argparse.ArgumentParser(description='Evaluation for nextgqa')
    parser.add_argument("--model_path", type=str, default="/path/to/qwen-model")
    parser.add_argument("--data_path", type=str, default="/path/to/data", required=True)
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--result_dir", type=str, default="./work_dirs_eval", help="Directory to save checkpoints")
    parser.add_argument("--num_gpus", type=int, default=8, help="GPU device to use")
    parser.add_argument("--fps", type=float, default=1.0, help="FPS of the video")
    parser.add_argument("--total_pixels", type=int, default=8192, help="Max new tokens")
    parser.add_argument("--prefix", type=str, default="base", help="Prefix of the result directory")
    return parser.parse_args()

def calc_iou(candidates, gt):
    start, end = candidates[:,0], candidates[:,1]
    s, e = gt[0], gt[1]
    inter = np.minimum(end, e) - np.maximum(start, s)
    union = np.maximum(end, e) - np.minimum(start, s)
    return inter.clip(min=0) / union

def create_work_items(data, video_root):
    examples = []
    for i, info in enumerate(data):
        video_path = os.path.join(video_root, info['video'])

        example = {
            "problem": {"question":info['question'], "options":info['options']},
            "solution": {"answer":info['answer'], "glue":info['glue']},
            "video_path": video_path,
            "durations": info['duration'],
            "video_id": i
        }

        examples.append(example)
    # # 随机打乱列表
    # random.shuffle(work_items)
    return examples

def setup_model(model_base, device):
    print(f"Setting up model on device {device}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_base,
        torch_dtype=torch.bfloat16,
        use_sliding_window=True,
        attn_implementation="flash_attention_2",
        device_map=device
    )
    processor = AutoProcessor.from_pretrained(model_base)
    return model, processor


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:" "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCDEFG]", s):
        return ""

    matches = re.search(r"[ABCDEFG]", s)
    if matches is None:
        return ""
    return matches[0]


def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = [list(i) for i in intervals] # tuple to list
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0][:]]
    for current in sorted_intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            merged[-1][1] = max(last[1], current[1])
        else:
            merged.append(current[:])
    
    return merged

def compute_iou(list_a, list_b):
    merged_a = merge_intervals(list_a)
    merged_b = merge_intervals(list_b)
    
    len_a = sum(end - start for start, end in merged_a)
    len_b = sum(end - start for start, end in merged_b)
    
    intersection = 0
    i = j = 0
    while i < len(merged_a) and j < len(merged_b):
        a_start, a_end = merged_a[i]
        b_start, b_end = merged_b[j]
        
        start = max(a_start, b_start)
        end = min(a_end, b_end)
        if start < end:
            intersection += end - start
        
        if a_end < b_end:
            i += 1
        else:
            j += 1
    
    union = len_a + len_b - intersection
    if union == 0:
        return 1.0
    
    return round(intersection / union, 4)

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

def inference(video_path, prompt, model, processor, max_new_tokens=512, device="cuda:0", timestamps=None, total_pixels=8192):
    messages = [
        {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"video": video_path, 
                "total_pixels": total_pixels * 28 * 28, 
                "min_pixels": 16 * 28 * 28,
                "fps": args.fps,
                "timestamps": timestamps,
                },
            ]
        },
    ]
    
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    fps_inputs = video_kwargs['fps']

    video_len = video_inputs[0].shape[0] / fps_inputs[0]
    video_prompt = "This video in total has " + str(round(video_len, 2)) + " seconds. "

    messages = [
        {"role": "user", "content": [
                {"type": "text", "text": video_prompt + prompt},
                {"video": video_path, 
                "total_pixels": total_pixels * 28 * 28, 
                "min_pixels": 16 * 28 * 28,
                "fps": args.fps,
                },
            ]
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, fps=fps_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True, return_dict_in_generate=True, output_logits=True)
    
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs.sequences[0][input_length:]
    output_text = processor.decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    answer_confidence, glue_confidence = confidence_score(torch.stack(outputs.logits)[:, 0], generated_tokens, processor)
    return output_text, round(answer_confidence, 2), round(glue_confidence, 2)

def process_work_items(work_items, model_base, device, result_dir):
    model, processor = setup_model(model_base, device)
    
    os.makedirs(f"{result_dir}/{model_base.split('/')[-1]}/NextGQA_{args.prefix}", exist_ok=True)
    log_path = f"{result_dir}/{model_base.split('/')[-1]}/NextGQA_{args.prefix}/{device}.json"
    print(log_path)
    pbar = tqdm(work_items)
    ious = []
    coarse_ious = []
    fine_ious = []
    accs = []
    accs_coarse = []
    accs_fine = []
    outputs = []
    for idx, item in enumerate(pbar):
        video_path = item['video_path']
        
        # prompt = item["problem"]["question"] + " " + ' '.join(item["problem"]["options"])
        prompt = "Answer the question: " + item["problem"]["question"] + " according to the content of the video. Select the answer from: " + ' '.join(item["problem"]["options"])

        try:
            coarse_ans, coarse_answer_confidence, coarse_glue_confidence = inference(video_path, prompt + ' ' + QUESTION_TEMPLATE, model, processor, device=device, total_pixels=args.total_pixels)
            print("ans", coarse_ans)

            pattern_answer = r'<answer>(.*?)</answer>'
            match_answer = re.search(pattern_answer, coarse_ans, re.DOTALL)

            coarse_acc = 0.0
            if match_answer:
                coarse_answer = match_answer.group(1)
                if extract_characters_regex(coarse_answer) == extract_characters_regex(item["solution"]["answer"]):
                    coarse_acc = 1.0


            pattern_glue = r'<glue>(.*?)</glue>'
            match_glue = re.search(pattern_glue, coarse_ans, re.DOTALL)

            pattern_think = r'<think>(.*?)</think>'
            match_think = re.search(pattern_think, coarse_ans, re.DOTALL)
            
            think_content = None
            if match_think:
                think_content = match_think.group(1).strip()
                think_content = re.sub(r'<time>.*?</time>', '', think_content)

            coarse_iou = 0.0
            pred_glue = None
            if match_glue:
                glue = match_glue.group(1).strip()
                if is_valid_two_d_list_format(glue):
                    pred_glue = ast.literal_eval(glue)
                    coarse_iou = compute_iou(pred_glue, item["solution"]["glue"])

            coarse_ious.append(coarse_iou)
            accs_coarse.append(coarse_acc)

            if pred_glue is not None:
                if think_content is not None:
                    direct_prompt = prompt + ' ' + QUESTION_TEMPLATE + " " + think_content
                else:
                    direct_prompt = prompt + ' ' + QUESTION_TEMPLATE
                pred_glue_extend = [[max(start - 10, 0), end + 10] for start, end in pred_glue]
                fine_ans, fine_answer_confidence, fine_glue_confidence = inference(video_path, direct_prompt, model, processor, device=device, timestamps=pred_glue_extend, total_pixels=4096)
            else:
                direct_prompt = prompt + ' ' + QUESTION_TEMPLATE
                fine_ans, fine_answer_confidence, fine_glue_confidence = inference(video_path, direct_prompt, model, processor, device=device, total_pixels=8192)
            
            match_answer = re.search(pattern_answer, fine_ans, re.DOTALL)
            fine_acc = 0.0
            if match_answer:
                fine_answer = match_answer.group(1)
                if extract_characters_regex(fine_answer) == extract_characters_regex(item["solution"]["answer"]):
                    fine_acc = 1.0

            accs_fine.append(fine_acc)
            match_glue = re.search(pattern_glue, fine_ans, re.DOTALL)
            fine_pred_glue = None
            fine_iou = 0.0
            if match_glue:
                glue = match_glue.group(1).strip()
                if is_valid_two_d_list_format(glue):
                    fine_pred_glue = ast.literal_eval(glue)
                    for glue_id in range(len(fine_pred_glue)):
                        fine_pred_glue[glue_id] = [fine_pred_glue[glue_id][0] + pred_glue_extend[0][0], fine_pred_glue[glue_id][1] + pred_glue_extend[0][0]]
                    if len(pred_glue) > 1:
                        fine_iou = compute_iou(pred_glue, item["solution"]["glue"])
                    else:
                        fine_iou = compute_iou(fine_pred_glue, item["solution"]["glue"])
            fine_ious.append(fine_iou)

            final_answer = fine_answer if fine_answer_confidence > coarse_answer_confidence else coarse_answer
            acc = 0.0
            if match_answer:
                fine_answer = match_answer.group(1)
                if extract_characters_regex(fine_answer) == extract_characters_regex(item["solution"]["answer"]):
                    acc = 1.0
            accs.append(acc)
            if fine_answer_confidence >= coarse_answer_confidence and fine_glue_confidence >= coarse_glue_confidence:
                iou = fine_iou
            else:
                iou = coarse_iou
            
            ious.append(iou)

            outputs.append({'video_path': video_path, 'prompt':prompt, 'gt':item["solution"], 'coarse_pred': coarse_ans, 'fine_pred': fine_ans, 'acc': acc, 'iou':iou, 'pred_glue':pred_glue, 'fine_acc': (fine_acc, fine_answer_confidence), 'coarse_acc': (coarse_acc, coarse_answer_confidence), 'fine_iou': (fine_iou, fine_glue_confidence), 'coarse_iou': (coarse_iou, coarse_glue_confidence)})
            print("outputs", outputs[-1])
            with open(log_path, 'w') as f:
                json.dump(outputs, f)
            
            pbar.set_postfix({"IoU": iou, "IoU_fine": fine_iou, "IoU_coarse": coarse_iou, 'acc': acc, 'acc_fine': fine_acc, 'acc_coarse': coarse_acc})
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
    
def evaluate(data, video_root, slurm_procid, args):
    work_items = create_work_items(data, video_root=video_root)
    
    process_work_items(
        work_items, 
        args.model_path, 
        f'cuda:{slurm_procid}', 
        f'{args.result_dir}',
    )

    dist.barrier()
    all_outputs = []
    for i in range(num_gpus):
        with open(f'{args.result_dir}/{args.model_path.split("/")[-1]}/NextGQA_{args.prefix}/cuda:{i}.json', 'r') as f:
            all_outputs.extend(json.load(f))
    
    # calculate iou and acc
    ious = [output['iou'] for output in all_outputs]
    coarse_ious = [output['coarse_iou'][0] for output in all_outputs]
    fine_ious = [output['fine_iou'][0] for output in all_outputs]
    accs = [output['acc'] for output in all_outputs]
    accs_fine = [output['fine_acc'][0] for output in all_outputs]
    accs_coarse = [output['coarse_acc'][0] for output in all_outputs]
    result = {"mIoU": sum(ious)/len(ious), "acc": sum(accs)/len(accs), "acc_fine": sum(accs_fine)/len(accs_fine), "acc_coarse": sum(accs_coarse)/len(accs_coarse)}
    result["mIoU_fine"] = sum(fine_ious)/len(fine_ious)
    result["mIoU_coarse"] = sum(coarse_ious)/len(coarse_ious)
    result["IoU@0.3"] = len([output['iou'] for output in all_outputs if output['iou'] > 0.3]) / len(all_outputs)
    result["IoU@0.5"] = len([output['iou'] for output in all_outputs if output['iou'] > 0.5]) / len(all_outputs)
    result["IoU@0.7"] = len([output['iou'] for output in all_outputs if output['iou'] > 0.7]) / len(all_outputs)
    
    with open(f'{args.result_dir}/{args.model_path.split("/")[-1]}/NextGQA_{args.prefix}/output.json', 'w') as f:
        json.dump(all_outputs, f)

    with open(f'{args.result_dir}/{args.model_path.split("/")[-1]}/NextGQA_{args.prefix}/result.json', 'w') as f:
        json.dump(result, f)

if __name__=='__main__':
    args = get_args()
    
    dist.init_process_group(backend="nccl")
    torch.distributed.barrier()
    world_size = torch.distributed.get_world_size()
    world_rank = torch.distributed.get_rank()
    num_gpus = args.num_gpus
    with open(f"{args.data_path}/nextgqa_test.json", "r") as f:
        data = json.load(f)
    data_chunks = split_data(data, num_gpus)
    current_data_chunk = data_chunks[world_rank]
    video_root = f"{args.data_path}/NExTVideo"
    
    evaluate(current_data_chunk, video_root, world_rank, args)
