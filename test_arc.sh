python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=tiiuae/falcon-40b-instruct,dtype=half,trust_remote_code=True \
    --num_fewshot 25 \
    --tasks arc_* \
    --output_path result_falcon_arc.json \
    --device cuda:1