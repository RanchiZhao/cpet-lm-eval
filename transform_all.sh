
# step=$1
for ((step=600;step<6000;step+=200))
do
  old_model_path=/data/checkpoints/ultrallama/ultrachat_llama-65b/step_${step}
  if [ -d $old_model_path ];then
    
    model_path=/data/checkpoints/ultrallama/ultrachat_llama-65b/step_${step}_hf
    echo "use model" $model_path

    if [ ! -d $model_path ];then
      echo "transform to huggingface, this may take a while"
      python -u transform_to_hf.py /data/checkpoints/ultrallama/ultrachat_llama-65b/step_${step}
    else
      echo $model_path "already exists, skip transforming"
    fi
  else
  continue
  fi
done


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
