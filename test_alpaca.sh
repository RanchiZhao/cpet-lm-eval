
conda create -n eval python=3.9
conda init bash
source /opt/conda/etc/profile.d/conda.sh
conda activate eval
pip install transformers==4.30.0
pip install sentencepiece
pip install protobuf==3.20.0
pip install einops
pip install -e .
pip install accelerate
pip install bitsandbytes
pip install sacrebleu

# model=$1
# gpu=$2
#  model_path=/data/checkpoints/ultrallama/ultrachat_llama-65b/step_${step}_hf
model_path=/mnt/data/user/tc_agi/user/chenyulin/models/alpaca-7b

echo "use model" $model_path

# if [ ! -d $model_path ];then
#   python -u transform_to_hf.py /data/checkpoints/ultrallama/ultrachat_llama-65b/step_${step}
#   else
#   echo $model_path "already exists"
# fi
names=(arc hellaswag truthfulqa mmlu)
datas=(arc_* hellaswag truthfulqa_mc hendrycksTest-*)
shots=(25 10 0 5)
mkdir -p /data/checkpoints/results

for ((i=3;i<4;i++))
do
    name=${names[i]}
    data=${datas[i]}
    shot=${shots[i]}
    echo "evaluate on ${data} shot=${shot}"
    result_path=/data/checkpoints/results/alpaca_${name}.json
    echo "result will be saved in" $result_path


    # echo "device1" $device1
    # echo "device2" $device2
    CMD="python -u main.py --model hf-causal-experimental --model_args pretrained=${model_path},dtype=half,trust_remote_code=True --num_fewshot $shot --tasks $data --output_path ${result_path} --batch_size auto"
    echo "-------Task ${i} final CMD is------"
    echo "${CMD}"
    echo "-------final CMD end------"
    $CMD
done
