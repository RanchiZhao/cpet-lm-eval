import torch, os
import json
from collections import OrderedDict
from tqdm import tqdm
import os

# inpath = f"/data/ultrallama/ultrallama-13b_reasoning_step5200_mc"
# outpath = f"/data/ultrallama/ultrallama-13b_reasoning_step5200_hf"
# os.makedirs(outpath, exist_ok=True)
# bmt_model = torch.load(os.path.join(inpath, "checkpoint.pt"))
def transform_to_hf(bmt_model, param_size):
    model_hf = OrderedDict()

    model_hf['model.embed_tokens.weight'] = bmt_model["input_embedding.weight"].contiguous().float()
    print(bmt_model["input_embedding.weight"].size())
    print("observe pad token embedding")
    print(bmt_model["input_embedding.weight"][-1])
    print("observe unn token embedding")
    print(bmt_model["input_embedding.weight"][0])
    model_hf['model.norm.weight'] = bmt_model["encoder.output_layernorm.weight"].contiguous().float()
    model_hf['lm_head.weight'] = bmt_model['output_projection.weight'].contiguous().float()

    if param_size == "13b":
        layernum = 40
    elif param_size == "65b":
        layernum = 80

    for lnum in range(layernum):
        hf_pfx = f"model.layers.{lnum}"
        bmt_pfx = f"encoder.layers.{lnum}"
        
        model_hf[f"{hf_pfx}.input_layernorm.weight"] = bmt_model[f"{bmt_pfx}.self_att.layernorm_before_attention.weight"].contiguous().float()

        model_hf[f"{hf_pfx}.self_attn.q_proj.weight"] = bmt_model[f"{bmt_pfx}.self_att.self_attention.project_q.weight"].contiguous().float()
        model_hf[f"{hf_pfx}.self_attn.k_proj.weight"] = bmt_model[f"{bmt_pfx}.self_att.self_attention.project_k.weight"].contiguous().float()
        model_hf[f"{hf_pfx}.self_attn.v_proj.weight"] = bmt_model[f"{bmt_pfx}.self_att.self_attention.project_v.weight"].contiguous().float()
        model_hf[f"{hf_pfx}.self_attn.o_proj.weight"] = bmt_model[f"{bmt_pfx}.self_att.self_attention.attention_out.weight"].contiguous().float()

        model_hf[f"{hf_pfx}.post_attention_layernorm.weight"] = bmt_model[f"{bmt_pfx}.ffn.layernorm_before_ffn.weight"].contiguous().float()

        model_hf[f"{hf_pfx}.mlp.gate_proj.weight"] = bmt_model[f"{bmt_pfx}.ffn.ffn.w_in.w_0.weight"].contiguous().float()
        model_hf[f"{hf_pfx}.mlp.up_proj.weight"] = bmt_model[f"{bmt_pfx}.ffn.ffn.w_in.w_1.weight"].contiguous().float()

        model_hf[f"{hf_pfx}.mlp.down_proj.weight"] = bmt_model[f"{bmt_pfx}.ffn.ffn.w_out.weight"].contiguous().float()
    return model_hf


if __name__ == "__main__":
    import sys
    import shutil
    in_path = sys.argv[-1]
    if "13b" in in_path:
        param_size = "13b"
    elif "65b" in in_path:
        param_size = "65b"
    else:
        raise ValueError(f"cannot detect param_size automatically from {in_path}")
    # in_path = "/data/checkpoints/ultrallama/ultrachat_llama-65b/step_600"
    out_path = in_path + "_hf"
    os.makedirs(out_path, exist_ok=True)
    if not os.path.exists(os.path.join(out_path, "pytorch_model.bin")):
        print("transforming...")
        hf_state_dict = transform_to_hf(torch.load(os.path.join(in_path, "checkpoint.pt")), param_size)
        print("done")
        torch.save(hf_state_dict, os.path.join(out_path, "pytorch_model.bin"))
    if param_size == "65b":
        base_dir = "/mnt/data/user/tc_agi/user/chenyulin/llama/llama-65b"
    elif param_size == "13b":
        # extra token in tokenizer
        base_dir = "/mnt/data/user/tc_agi/user/chenyulin/ultrallama"
    else:
        raise NotImplementedError
    for n in ["config.json", "generation_config.json", "tokenizer_config.json", "added_tokens.json", "special_tokens_map.json", "tokenizer.model"]:
        if os.path.exists(os.path.join(base_dir, n)):
            shutil.copy(os.path.join(base_dir, n), os.path.join(out_path, n))
    print("saved")
    print(list(os.listdir(out_path)))




