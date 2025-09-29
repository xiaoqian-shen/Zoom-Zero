NAME="CGBench NextGQA ReXTime"
full_model_path=$1
base_name=$(basename "$full_model_path")
OUTPUT_DIR=work_dirs_eval/$base_name
prefix="gqa"

if [[ "$NAME" == *"NextGQA"* ]]; then
    bash examples/eval_scripts/eval_nextgqa.sh $full_model_path $prefix ./dataset/NExT-GQA
fi

if [[ "$NAME" == *"ReXTime"* ]]; then
    bash examples/eval_scripts/eval_rextime.sh $full_model_path $prefix ./dataset/ReXTime
fi

if [[ "$NAME" == *"CGBench"* ]]; then
    bash examples/eval_scripts/eval_cgbench.sh $full_model_path $prefix ./dataset/CG-Bench
fi