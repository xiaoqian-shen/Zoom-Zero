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

import math
import os
import yaml
import json
import ast
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import random
import torch
from datasets import load_dataset
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from ..utils.vision_process import process_vision_info, fetch_video, smart_resize
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF

QUESTION_TEMPLATE_VTG = """
Provide the interval in seconds within <glue> </glue> tags. For example: <glue>[(20.3, 30.8)]</glue>.
"""

QUESTION_TEMPLATE = """
Identify the key visual content relevant to the given question and options, marking precise timestamps or time ranges in seconds within <time> </time> tags, and present them in an interleaved analysis format. Enclose the full analysis in <think> </think> tags. For example: <think> After folding the face towel <time>(5.2, 10.4)</time>, the person placed it on the bed <time>(20.3, 30.8)</time>.</think>
Then, provide your answer within the <answer> </answer> tags, output the corresponding letter of the option. At the same time, in the <glue> </glue> tags, include only the precise video segments (in seconds) that strongly support your answer, in the format of [(s1, e1), (s2, e2), ...]. Do not list unrelated time ranges. For example: <answer>A</answer>\n<glue>[(20.3, 30.8)]</glue>.
"""

def resize_video(video, sample_fps, total_pixels=4096*28*28, min_pixels=16*28*28, image_factor: int = 28):
    maximum_frames = int(total_pixels / (min_pixels * 1.05) * 2)
    if video.shape[0] > maximum_frames:
        frame_idx = torch.linspace(0, video.shape[0] - 1, maximum_frames).round().long().tolist()
        sample_fps = maximum_frames / video.shape[0] * sample_fps
        video = video[frame_idx]

    nframes, _, height, width = video.shape
    
    max_pixels = max(min(768 * 28 * 28, total_pixels / nframes * 2.0), int(min_pixels * 1.05))
    max_pixels = min(total_pixels, max_pixels)
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=image_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    video = transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()
    return video, sample_fps

def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(image: Union[Dict[str, Any], ImageObject], max_pixels: int, min_pixels: int) -> ImageObject:
    if isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "video",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        system_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        config: Optional[dict] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.system_prompt = system_prompt
        self.config = config

        self.dataset = []
        self.data_folders = {}

        with open(data_path, "r") as file:
            yaml_data = yaml.safe_load(file)
            datasets = yaml_data.get("datasets")
            for dataset in datasets:
                json_path = dataset.get("json_path")
                with open(json_path, "r") as json_file:
                    cur_data_dict = json.load(json_file)
                if "plm" in json_path:
                    source = "plm"
                elif "nextgqa" in json_path:
                    source = "nextgqa"
                elif "qvhighlights" in json_path:
                    source = "qvhighlights"
                elif "anet" in json_path:
                    source = "anet"
                elif "ego4d" in json_path:
                    source = "ego4d"
                elif "coin" in json_path:
                    source = "coin"
                elif "hirest" in json_path:
                    source = "hirest"
                elif "medvidqa" in json_path:
                    source = "medvidqa"
                elif "tacos" in json_path:
                    source = "tacos"
                elif "youcook2" in json_path:
                    source = "youcook2"
                elif "perceptiontest" in json_path:
                    source = "perceptiontest"
                elif "charades" in json_path:
                    source = "charades"
                else:
                    source = "gqa"
                for idx, data_item in enumerate(cur_data_dict):
                    cur_data_dict[idx]['dataset'] = source
                
                self.dataset.extend(cur_data_dict)

                self.data_folders[source] = dataset["data_folder"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset[index]
        source = example['dataset']
        data_dict = {}

        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example.pop(self.image_key)
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"images": images}

        elif "video" in example:
            video_name = example['video'] if ".mp4" in example['video'] else example['video'] + ".mp4"
            if "ego4d" in source:
                video_name = example['video'] + ".npy"
            video_path = os.path.join(self.data_folders[source], video_name)
            if self.config.task == "gqa":
                choices = ' '.join(example["options"])
                problem = "Answer the question: " + example["question"] + " according to the content of the video. Select the answer from:  " + choices
            else:
                problem = example["question"]
            
            video_inputs_raw, raw_fps_inputs = fetch_video({"video": video_path, "fps": self.video_fps}, return_video_sample_fps=True, resize=False)

            video_inputs, fps_inputs = resize_video(video_inputs_raw, raw_fps_inputs, self.max_pixels, self.min_pixels)
            nframe_inputs = video_inputs.shape[0]

            length_inputs = round(nframe_inputs / fps_inputs, 2)
            system_prompt = QUESTION_TEMPLATE if self.config.task == "gqa" else QUESTION_TEMPLATE_VTG
            video_prompt = "This video in total has " + str(length_inputs) + " seconds. "

            messages = [
                {"role": "user", "content": [
                    {
                        "type": "video", 
                        "total_pixels": self.max_pixels, 
                        "min_pixels": self.min_pixels,
                        "video": video_path,
                        "fps": self.video_fps,
                    },
                    {
                        "type": "text",
                        "text": video_prompt + problem + '\n' + system_prompt
                    },
                ]},
            ]
            
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            model_inputs = self.processor(
                videos=video_inputs, text=prompt, fps=fps_inputs, add_special_tokens=False, return_tensors="pt"
            )

            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            data_dict["multi_modal_data"] = {"video": video_inputs}
            data_dict["mm_processor_kwargs"] = {'fps': fps_inputs}
            data_dict["multi_modal_inputs"] = dict(model_inputs)
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        # if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen2vl mrope
        position_ids = get_rope_index(
            self.processor,
            input_ids=input_ids,
            video_grid_thw=model_inputs["video_grid_thw"],
            second_per_grid_ts=model_inputs["second_per_grid_ts"],
            attention_mask=attention_mask,
        )  # (3, seq_length)
        # else:
        #     position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        if self.config.zoom:
            raw_question = problem + ". " + "Respond with only the letter of the option."
            raw_question = self.tokenizer([raw_question], add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            data_dict["raw_video_inputs"] = {"video": np.array(video_inputs_raw.unsqueeze(0)), "fps": np.array(raw_fps_inputs), "raw_question": np.array(raw_question)}
        
        if isinstance(example["glue"], str):
            example["glue"] = ast.literal_eval(example["glue"])
        
        data_dict["input_ids"] = input_ids
        data_dict["attention_mask"] = attention_mask
        data_dict["position_ids"] = position_ids
        data_dict["raw_prompt_ids"] = raw_prompt_ids
        data_dict["ground_truth"] = example['answer'].replace("(", "").replace(")", "") if self.config.task == "gqa" else "C"
        data_dict["gt_frame"] = {"glue": np.array(example["glue"])}
        data_dict["video_length"] = length_inputs
        return data_dict

from .tokenizer import get_processor, get_tokenizer

if __name__ == "__main__":
    model_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/lihongyu/Qwen/Qwen2.5-VL-3B-Instruct"
    data_path = '/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mtcv/lihongyu/projects/video_llm/codes/VLM-R1/src/EasyR1/scripts/tvg.yaml'
    tokenizer = get_tokenizer(model_path)
    processor = get_processor(model_path, use_fast=False)

    dataset = RLHFDataset(
        data_path,
        tokenizer,
        processor,
        min_pixels=2592,
        max_pixels=2592*2
        # max_pixels=14 * 14 * 1024 * 8,
    )
    # Qwen-VL patch size 2 * 14 * 14
    for data in dataset:
        pass
