SAVE_NAME=llama-7b


# step=$1
PET_PATH=/mnt/data/user/tc_agi/user/huangyuxiang/alpaca-lora
# PET_PATH=/data/checkpoints/models
# echo $step

i=3

names=(arc hellaswag truthfulqa mmlu)
datas=(arc_* hellaswag truthfulqa_mc hendrycksTest-*)
shots=(25 10 0 5)
name=${names[i]}
data=${datas[i]}
shot=${shots[i]}

echo "task" $data
echo "shot" $shot

conda deactivate
conda remove -n eval
conda create -n eval python=3.9
conda init bash
source /opt/conda/etc/profile.d/conda.sh
conda activate eval
pip install bmtrain-zh==0.2.3.dev9
pip install jieba
pip install transformers==4.31.0
pip install sentencepiece
pip install protobuf==3.20.0
pip install einops
pip install -e .
pip install accelerate
pip install bitsandbytes


model_path=/mnt/data/user/tc_agi/user/zhaoweilin/llama-7b
# model_path=/mnt/data/user/tc_agi/ws/hf-13B
echo "use model" $model_path

# if [ ! -d $model_path ];then
# python -u transform_to_hf.py /data/checkpoints/ultrallama/ultrachat_llama-65b-resumed-part1-new/step_${step}
  #else
  #echo $model_path "already exists"
#fi


result_path=/data/checkpoints/logs/
# mkdir -p ${result_path}
echo "result will be saved in" $result_path


# echo "device1" $device1
# echo "device2" $device2
CMD="python -u main.py --model modelcenter-llama-ft --model_args pretrained=${model_path},dtype="float16",use_accelerate=False --num_fewshot $shot --tasks $data --batch_size auto --output_path ${result_path}/result_${name}.json"
echo "-------Task ${i} final CMD is------"
echo "${CMD}"
echo "-------final CMD end------"
${CMD} 2>&1 | tee /data/checkpoints/logs/${SAVE_NAME}.log
# done
# wait
# python main.py \
#     --model hf-causal \
#     --model_args pretrained=/data/ultrallama/ultrallama-13b_reasoning_step3400_hf \
#     --num_fewshot 0 \
#     --tasks truthfulqa_mc \
#     --output_path result_truthfulqa.json \
#     --device cuda:1

# --tasks hellaswag \
# ,arc_easy,arc_challenge
# hellaswag,truthfulqa_mc
