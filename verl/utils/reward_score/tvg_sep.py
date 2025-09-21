import re
import json
import ast
import numpy as np
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

def merge_intervals(intervals):
    if intervals is None or len(intervals) == 0:
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

def compute_iou(list_a, list_b, video_length, div_union=True):
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
    if union == 0 or len_b == 0:
        return 0.0
    if div_union:
        return intersection / union
    else:
        return intersection / len_b

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

def iou_reward(predict_str: str, ground_truth: list, video_length: float) -> float:
    pattern_glue = r'<glue>(.*?)</glue>'
    match_glue = re.search(pattern_glue, predict_str, re.DOTALL)
    pred_glues = []
    if match_glue:
        glue = match_glue.group(1).strip()
        if is_valid_two_d_list_format(glue):
            pred_glues = ast.literal_eval(glue)
        reward = compute_iou(pred_glues, ground_truth.tolist(), video_length)
    else:
        reward = 0.0
    return reward


def answer_reward(predict_str: str, ground_truth: str) -> float:
    predict_str = predict_str.split("</think>")[-1]
    pattern_answer = r'<answer>(.*?)</answer>'
    match_answer = re.search(pattern_answer, predict_str, re.DOTALL)
    reward = 0.0
    if match_answer:
        answer = match_answer.group(1).strip().replace("(", "").replace(")", "")
        if answer == ground_truth:
            reward = 1.0
    return reward

def tvg_format_reward(predict_str: str) -> float:
    pattern = re.compile(
        r'<think>.*?<time>.*?</time>.*?</think>\s*\n'              # reasoning block with <time> inside
        r'<answer>\(?[A-F]\)?</answer>\s*\n'                       # answer (A-F)
        r'<glue>\s*\[\s*\(\d+\.?\d*,\s*\d+\.?\d*\)'              # first timestamp
        r'(?:\s*,\s*\(\d+\.?\d*,\s*\d+\.?\d*\))*\s*\]\s*</glue>',# optional more timestamps
        # r'\s*<conf>.*?</conf>\s*',                               # confidence
        re.DOTALL
    )
    pattern_time = re.compile(r'<time>(.*?)</time>')
    pattern_glue = r'<glue>(.*?)</glue>'
    format_match = re.fullmatch(pattern, predict_str.strip())
    reward = 0.0
    if format_match:
        reward = 1.0
        match_glue = re.search(pattern_glue, predict_str.strip(), re.DOTALL)
        if match_glue:
            glue = match_glue.group(1).strip()
            if not is_valid_two_d_list_format(glue):
                reward -= 0.5
    
    reward = max(reward, 0.0)

    conf_pattern = re.compile(r'<conf>(.*?)</conf>', re.DOTALL)
    conf_match = conf_pattern.search(predict_str)
    if conf_match:
        conf = conf_match.group(1)
        try:
            conf = float(conf)
        except:
            conf = 0.5
    else:
        conf = 0.5
    return reward, conf

def iou_think_reward(predict_str: str, ground_truth: list, video_length: float) -> float:
    pattern_glue = re.compile(r'<time>(.*?)</time>')
    timestamps = pattern_glue.findall(predict_str)
    if timestamps:
        pred_glues = []
        for timestamp in timestamps:
            timestamp = "[" + timestamp.strip() + "]"
            if is_valid_two_d_list_format(timestamp):
                pred_glues.extend(ast.literal_eval(timestamp))
        reward = compute_iou(pred_glues, ground_truth.tolist(), video_length, div_union=False)
    else:
        reward = 0.0
    return reward

def tvg_sep_compute_score(predict_str: str, ground_truth: list, gt_frame: list, video_length: float) -> float:
    tg_reward = iou_reward(predict_str, gt_frame, video_length)
    tg_think_reward = iou_think_reward(predict_str, gt_frame, video_length)
    acc_reward = answer_reward(predict_str, ground_truth)
    format_reward, conf = tvg_format_reward(predict_str)
    reward = (acc_reward + format_reward + tg_reward) / 3.0
    scores = {
        "iou": tg_reward,
        "answer": acc_reward,
        "format": format_reward,
        "conf": conf,
        "zoom": 0.0
    }
    return reward, scores
