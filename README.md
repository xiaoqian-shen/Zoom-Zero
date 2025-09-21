# Zoom-Zero

## 🚀 Training

The list of training data is shown below.

| Dataset | Source |
|-|:-:|
| NExT-GQA| [Download](https://huggingface.co/datasets/jinyoungkim/NExT-GQA) |
| ActivityNet | [Download](https://huggingface.co/datasets/WHB139426/Grounded-VideoLLM/tree/main/activitynet) |
| QVhighlight | [Download](https://huggingface.co/datasets/WHB139426/Grounded-VideoLLM/tree/main/qvhighlights) |
| PLM-Video | [Download](https://huggingface.co/datasets/facebook/PLM-Video-Auto) |

```
# Training with 8 80G-A100
bash example/train_scripts/train.sh
```

## 📝 Evaluation

The list of benchmarks is shown below.

| Dataset | Task |
|-|:-:|
| [NExT-GQA](https://huggingface.co/datasets/jinyoungkim/NExT-GQA) | Grounded VideoQA |  
| [ReXTime](https://huggingface.co/datasets/ReXTime/ReXTime) | Grounded VideoQA |  
| [CG-Bench](https://huggingface.co/datasets/CG-Bench/CG-Bench) | Grounded VideoQA |
| [Video-MME](https://huggingface.co/datasets/lmms-lab/Video-MME) | Long VideoQA |
| [MLVU](https://huggingface.co/datasets/MLVU/MVLU) | Long VideoQA |
| [LVBench](https://huggingface.co/datasets/zai-org/LVBench) | Long VideoQA |

Here we provide the script for running the evaluation.

```
# Evaluate all benchmarks
bash example/eval_scripts/eval_all.sh MODEL_PATH
```
