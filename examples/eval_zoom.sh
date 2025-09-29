NAME="CGBench VideoMME LVBench MLVU"
full_model_path=$1
base_name=$(basename "$full_model_path")
OUTPUT_DIR=work_dirs_eval/$base_name
prefix="zoom"
chunk_size=$2

if [[ "$NAME" == *"VideoMME"* ]]; then
    bash examples/eval_scripts/eval_videomme_zoom.sh $full_model_path $prefix /home/xiaoqians/personal/dataset/Video-MME $chunk_size
fi

if [[ "$NAME" == *"MLVU"* ]]; then
    bash examples/eval_scripts/eval_mlvu_zoom.sh $full_model_path $prefix /home/xiaoqians/personal/dataset/MLVU/MLVU $chunk_size
fi

if [[ "$NAME" == *"LVBench"* ]]; then
    bash examples/eval_scripts/eval_lvb_zoom.sh $full_model_path $prefix /home/xiaoqians/personal/dataset/LVBench $chunk_size
fi

if [[ "$NAME" == *"CGBench"* ]]; then
    bash examples/eval_scripts/eval_cgbench_zoom.sh $full_model_path $prefix /home/xiaoqians/personal/dataset/CG-Bench $chunk_size
fi