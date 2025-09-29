NAME="MLVU VideoMME LVBench"
full_model_path=$1
base_name=$(basename "$full_model_path")
OUTPUT_DIR=work_dirs_eval/$base_name
prefix="mcq"

if [[ "$NAME" == *"VideoMME"* ]]; then
    bash examples/eval_scripts/eval_videomme.sh $full_model_path ${prefix} ./dataset/Video-MME
    bash examples/eval_scripts/eval_videomme.sh $full_model_path ${prefix}_sub ./dataset/Video-MME
fi

if [[ "$NAME" == *"MLVU"* ]]; then
    bash examples/eval_scripts/eval_mlvu.sh $full_model_path $prefix ./dataset/MLVU/MLVU
fi

if [[ "$NAME" == *"LVBench"* ]]; then
    bash examples/eval_scripts/eval_lvb.sh $full_model_path $prefix ./dataset/LVBench
fi