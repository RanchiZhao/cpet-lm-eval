# mkdir -p /data/checkpoints/models
# echo "start copying ckpt ..."
# cp /data/checkpoints/results/Alpaca-finetune/finetune-llama-Alpaca-9.pt /data/checkpoints/models
# mv /data/checkpoints/models/finetune-llama-Alpaca-9.pt /data/checkpoints/models/checkpoint.pt
# echo "finish copying ckpt"
# echo "start transforming ..."
# python transform_to_hf_lora.py /data/checkpoints/models/
# echo "finish transforming"
hdfs dfs -put /data/checkpoints/models/adapter_model.bin user/huangyuxiang/alpaca-lora
