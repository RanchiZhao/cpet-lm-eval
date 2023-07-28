

step=$1
echo $step

# i=$2


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


# basedir="/data/checkpoints/ultrallama/ultrachat_llama-65b-part1-fromscratch"
basedir="/data/checkpoints/ultrallama/llama-2-70b"


# while true; do


model_path="${basedir}/step_${step}_hf"
result_path="${model_path}"

# if [[ ! -e "${model_path}/eval_*.json" ]]; then
# echo "Transforming model ${step} to Hugging Face format..."
# python -u transform_to_hf.py "${basedir}/step_${step}"
# echo "Transformation done"
echo "*******************Evaluating model ${model_path}*******************"

echo "Running test on model ${model_path}"
cmd="python -u test_full.py --model hf-causal-experimental --model_args pretrained=${model_path},dtype=\"float16\",use_accelerate=True --batch_size auto --output_path ${result_path}"
echo "-------Task final CMD is------"
echo "${cmd}"
echo "-------final CMD end------"
${cmd}
echo "************************ Evaluation done, results saved at ${result_path} *********************"
#     fi

#     sleep 600  # Wait for 60 seconds before checking for the latest model again
# done





