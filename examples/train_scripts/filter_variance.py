import json
import os
import re
import ast
import numpy as np
from tqdm import tqdm

def interval_intersection(interval1, interval2):
    try:
        start = max(interval1[0], interval2[0])
        end = min(interval1[1], interval2[1])
        return max(0, end - start)
    except:
        return 0

def merge_time_ranges(ranges):
    if ranges is None or len(ranges) == 0:
        return []
    sorted_ranges = sorted(ranges, key=lambda x: x[0])
    merged = [sorted_ranges[0]]
    for current_start, current_end in sorted_ranges[1:]:
        last_merged_start, last_merged_end = merged[-1]
        if current_start <= last_merged_end:
            merged[-1] = (last_merged_start, max(last_merged_end, current_end))
        else:
            merged.append((current_start, current_end))
    return merged

def compute_iou(list_a, list_b):
    list_a = merge_time_ranges(list_a)
    list_b = merge_time_ranges(list_b)
    intersection = sum(interval_intersection(interval_a, interval_b) for interval_a in list_a for interval_b in list_b)
    total_pred = sum(p[1] - p[0] for p in list_a)
    total_gt = sum(g[1] - g[0] for g in list_b)
    union = total_pred + total_gt - intersection
    if union == 0:
        return 0
    return intersection / union

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

root_path = ""
all_output = []
if os.path.exists(os.path.join(root_path, "output.json")):
    with open(os.path.join(root_path, "output.json"), "r") as f:
        all_output = json.load(f)
else:
    for file in os.listdir(root_path):
        try:
            if file.endswith(".json") and file.startswith("cuda"):
                with open(os.path.join(root_path, file), "r") as f:
                    data = json.load(f)
                all_output.extend(data)
        except:
            print(file)

    result_filename = os.path.join(root_path, f"output.json")
    with open(result_filename, "w") as f:
        json.dump(all_output, f)


save_items = []
pattern_answer = r'<answer>(.*?)</answer>'
pattern_glue = r'<glue>(.*?)</glue>'
max_iou_distribution = {}
delta_iou_distribution = {}
for iou in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
    max_iou_distribution[iou] = 0
for iou in [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
    delta_iou_distribution[iou] = 0
durations = []
for item in tqdm(all_output):
    pred_answer_n = []
    pred_glues_n = []
    gt_answer = item['answer'].replace("(", "").replace(")", "")
    gt_glue = item['glue']
    for predict_str in item['pred']:
        match_answer = re.search(pattern_answer, predict_str, re.DOTALL)
        if match_answer:
            pred_answer = match_answer.group(1)
        else:
            pred_answer = ""
        pred_answer_n.append(pred_answer)

        match_glue = re.search(pattern_glue, predict_str, re.DOTALL)
        pred_glues = [[]]
        if match_glue:
            glue = match_glue.group(1).strip()
            if is_valid_two_d_list_format(glue):
                pred_glues = ast.literal_eval(glue)
        pred_glues_n.append(pred_glues)
    ious = []
    for pred_glue in pred_glues_n:
        try:
            iou = compute_iou(pred_glue, gt_glue)
        except:
            iou = 0.0
        ious.append(iou)
    max_iou = np.max(ious)
    delta_iou = np.max(ious) - np.mean(ious)
    hit = np.mean([1 if gt_answer in pred_answer else 0 for pred_answer in pred_answer_n])
    max_iou_distribution[np.floor(max_iou * 10) / 10] += 1
    delta_iou_distribution[np.floor(delta_iou * 10) / 10] += 1
    if (delta_iou > 0.1 or np.min(ious) > 0.0) and 0.0 <= hit < 1.0:
        item.pop('pred')
        save_items.append(item)
        durations.append(item['duration'])