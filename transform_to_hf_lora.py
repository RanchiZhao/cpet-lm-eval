import torch, os
import json
from collections import OrderedDict
from tqdm import tqdm
import os

def transform_to_hf(bmt_model):
    model_hf = OrderedDict()
    layernum = 32

    for lnum in range(layernum):
        
        hf_pfx = f"base_model.model.model.layers.{lnum}.self_attn"
        bmt_pfx = f"encoder.layers.{lnum}.self_att.self_attention"
        
        model_hf[f"{hf_pfx}.q_proj.lora_A.weight"] = bmt_model[f"{bmt_model}.project_q.lora.lora_A"].contiguous().float()
        model_hf[f"{hf_pfx}.k_proj.lora_A.weight"] = bmt_model[f"{bmt_model}.project_k.lora.lora_A"].contiguous().float()
        model_hf[f"{hf_pfx}.v_proj.lora_A.weight"] = bmt_model[f"{bmt_model}.project_v.lora.lora_A"].contiguous().float()
        model_hf[f"{hf_pfx}.o_proj.lora_A.weight"] = bmt_model[f"{bmt_model}.attention_out.lora.lora_A"].contiguous().float()
        model_hf[f"{hf_pfx}.q_proj.lora_B.weight"] = bmt_model[f"{bmt_model}.project_q.lora.lora_B"].contiguous().float()
        model_hf[f"{hf_pfx}.k_proj.lora_B.weight"] = bmt_model[f"{bmt_model}.project_k.lora.lora_B"].contiguous().float()
        model_hf[f"{hf_pfx}.v_proj.lora_B.weight"] = bmt_model[f"{bmt_model}.project_v.lora.lora_B"].contiguous().float()
        model_hf[f"{hf_pfx}.o_proj.lora_B.weight"] = bmt_model[f"{bmt_model}.attention_out.lora.lora_B"].contiguous().float()
        
    return model_hf


if __name__ == "__main__":
    import sys
    import shutil
    in_path = sys.argv[-1]
    
    out_path = in_path + "_hf"
    os.makedirs(out_path, exist_ok=True)
    if not os.path.exists(os.path.join(out_path, "adapter_model.bin")):
        print("transforming...")
        hf_state_dict = transform_to_hf(torch.load(os.path.join(in_path, "checkpoint.pt")))
        print("done")
        torch.save(hf_state_dict, os.path.join(out_path, "adapter_model.bin"))
        
    print("saved")
    print(list(os.listdir(out_path)))




