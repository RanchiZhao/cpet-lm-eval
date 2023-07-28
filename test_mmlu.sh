python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=tiiuae/falcon-40b-instruct,dtype="float16",trust_remote_code=True,use_accelerate=True \
    --num_fewshot 5 \
    --tasks hendrycksTest-* \
    --output_path result_falcon_mmlu.json \
    --device cuda:0,cuda:1