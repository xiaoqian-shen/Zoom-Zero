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


import torch
from transformers import PreTrainedTokenizer
from collections import defaultdict

from ... import DataProto
from ...utils.reward_score import math_compute_score, r1v_compute_score, tvg_compute_score, tvg_val_compute_score, tvg_zoom_compute_score

def find_sublist_indices(lst, sublst):
    """Return all start indices where sublst is found in lst."""
    n, m = len(lst), len(sublst)
    indices = []
    for i in range(n - m + 1):
        if lst[i:i+m] == sublst:
            indices.append(i)
            break
    return indices

class CustomRewardManager:
    def __init__(self, tokenizer: PreTrainedTokenizer, num_examine: int, compute_score: str):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_type = compute_score
        if self.reward_type == "math":
            self.compute_score = math_compute_score
        elif self.reward_type == "r1v":
            self.compute_score = r1v_compute_score
        elif self.reward_type == "tvg":
            self.compute_score = tvg_compute_score
        elif self.reward_type == "tvg_val":
            self.compute_score = tvg_val_compute_score
        elif self.reward_type == "tvg_zoom":
            self.compute_score = tvg_zoom_compute_score
        else:
            raise NotImplementedError()

    def __call__(self, data: DataProto) -> torch.Tensor:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_dict = defaultdict()
        for key in ["answer", "format", "iou", "zoom", "mask_iou", "mask_answer"]:
            reward_dict[key] = torch.zeros(len(data), dtype=torch.float32)
        already_print = 0

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            if "responses_zoom" in data_item.batch:
                response_ids_zoom = data_item.batch["responses_zoom"]
                response_str_zoom = self.tokenizer.decode(response_ids_zoom, skip_special_tokens=True)

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            glue_start_loc = find_sublist_indices(valid_response_ids.tolist(), [27, 6072])
            answer_start_loc = find_sublist_indices(valid_response_ids.tolist(), [27, 9217])

            ground_truth = data_item.non_tensor_batch["ground_truth"]
            gt_frame = data_item.non_tensor_batch["gt_frame"]["glue"]
            video_length = data_item.non_tensor_batch["video_length"]
            if "responses_zoom" in data_item.batch:
                score, scores_dict = self.compute_score(response_str, ground_truth, gt_frame, video_length, response_str_zoom)
            else:
                score, scores_dict = self.compute_score(response_str, ground_truth, gt_frame, video_length)
            for key in scores_dict.keys():
                if key != "mask_iou" and key != "mask_answer":
                    reward_dict[key][i] = scores_dict[key]
            reward_dict["mask_iou"][i] = glue_start_loc[0] if len(glue_start_loc) > 0 else valid_response_length - 1
            reward_dict["mask_answer"][i] = max(0, reward_dict["mask_iou"][i] - 5)
            reward_tensor[i, valid_response_length - 1] = score

            if already_print < self.num_examine:
                already_print += 1
                # print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[gt_frame]", gt_frame)
                print("[IoU]", scores_dict["iou"])
                print("[format]", scores_dict["format"])
                print("[answer]", scores_dict["answer"])
                print("[score]", score)
                if "responses_zoom" in data_item.batch:
                    print("[response_zoom]", response_str_zoom)

        return reward_tensor, reward_dict
