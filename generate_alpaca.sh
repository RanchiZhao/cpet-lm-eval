param_size=$1
step=$2

pip install transformers==4.28.1
pip install sentencepiece
pip install protobuf==3.20.0
pip install einops
pip install accelerate

if [ $param_size -eq 65 ]; then
    python -u generate_alpaca.py /data/checkpoints/ultrallama/ultrachat_llama-65b/step_${step}_hf
else
    python -u generate_alpaca.py /data/checkpoints/ultrallama/ultrachat_llama-13b-modelcenter-fulldata/step_${step}_hf
fi
