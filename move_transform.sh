mkdir -p /data/checkpoints/models
cp /data/checkpoints/results/Alpaca-finetune/finetune-llama-Alpaca-9.pt /data/checkpoints/models
mv /data/checkpoints/models/finetune-llama-Alpaca-9.pt /data/checkpoints/models/checkpoint.pt
python transform_to_hf_lora.py /data/checkpoints/models/
