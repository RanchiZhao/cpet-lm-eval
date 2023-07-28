
# step=$1
# echo $step

# i=$2

names=(arc_challenge hellaswag truthfulqa mmlu)
datas=(arc_challenge_sample hellaswag_sample truthfulqa_mc_sample hendrycksTest-sample-*)
shots=(25 10 0 5)


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

# basedir=/data/checkpoints/ultrallama/ultrachat_llama-65b-part1-fromscratch
# step=400

# old_model_path=${basedir}/step_${step}

# echo "transform to huggingface, this may take a while"
# python -u transform_to_hf.py ${basedir}/step_${step}
# echo "transformation done"

# model_path=${basedir}/step_${step}_hf

# result_path=${model_path}

# CMD="python -u test_sample.py --model hf-causal-experimental --model_args pretrained=${model_path},dtype="float16",use_accelerate=True  --batch_size auto --output_path ${result_path}"
# echo "-------Task ${i} final CMD is------"
# echo "${CMD}"
# echo "-------final CMD end------"
# $CMD


basedir="/data/checkpoints/ultrallama/ultrachat_llama-65b-part1-fromscratch"


while true; do
    latest_model=$(ls -d ${basedir}/step_* | grep -Eo "[0-9]+$" | sort -nr | head -n 1)
    echo "***************Detect latest step ${latest_model}****************"
    model_path="${basedir}/step_${latest_model}_hf"
    result_path="${model_path}"

    if [[ ! -e "${model_path}/eval_*.json" ]]; then
        echo "*******************Evaluating latest model ${model_path}*******************"
        echo "Transforming model ${latest_model} to Hugging Face format..."
        python -u transform_to_hf.py "${basedir}/step_${latest_model}"
        echo "Transformation done"

        echo "Running test on model ${model_path}"
        cmd="python -u test_sample.py --model hf-causal-experimental --model_args pretrained=${model_path},dtype=\"float16\",use_accelerate=True --batch_size auto --output_path ${result_path}"
        echo "-------Task final CMD is------"
        echo "${cmd}"
        echo "-------final CMD end------"
        ${cmd}
        echo "************************ Evaluation done, results saved at ${result_path} *********************"
    fi

    sleep 600  # Wait for 60 seconds before checking for the latest model again
done





