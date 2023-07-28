param_size=$1
step=$2
if [ $param_size -eq 65 ]; then
    # python -u transform_to_hf.py /data/checkpoints/ultrallama/ultrachat_llama-65b/step_${step}
    python -u transform_to_hf.py /data/checkpoints/ultrallama/ultrachat_llama-65b-resumed-part1-new/step_${step}
else
    python -u transform_to_hf.py /data/checkpoints/ultrallama/ultrachat_llama-13b-modelcenter-fulldata/step_${step}
fi
