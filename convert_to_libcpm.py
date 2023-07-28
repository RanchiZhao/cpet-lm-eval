
import torch
import struct
import numpy as np
import math
import sys
import os

def write_string(fp, v):
    v = v.encode("utf-8")
    fp.write( struct.pack("I", len(v)) )
    fp.write(v)

def write_tuple(fp, v):
    fp.write( struct.pack("B", len(v)) )
    for i in v:
        fp.write( struct.pack("I", i) )

def write_dtype(fp, v):
    sv = -1
    if v == np.int8:
        sv = 0
    elif v == np.float16:
        sv = 1
    if sv == -1:
        raise TypeError("Unknown dtype %s" % v)
    fp.write( struct.pack("B", sv) )

def write_parameter(fp, name : str, value : torch.Tensor):
    write_string(fp, name)
    write_tuple(fp, value.size())
    value = np.ascontiguousarray(value.cpu().numpy())
    value_bytes = value.tobytes()
    fp.write( struct.pack("I", len(value_bytes)) )
    write_dtype(fp, value.dtype)
    fp.write(value_bytes)

def split(x, s):
    sizes = []
    for it in x.size():
        sizes.append(it)
    assert sizes[0] % s == 0
    sizes = [s, sizes[0] // s ] + sizes[1:]
    return x.reshape(*sizes)


def convert_model(step, num_layers, load_gpu=0):
    model = {}
    model = torch.load(f"/data/checkpoints/ultrallama/llama-2-70b/step_{step}_hf/pytorch_model.bin")
    for k in model.keys():
        model[k] = model[k].half()

    params = {}

    params["token_embedding.weight"] =(model["model.embed_tokens.weight"]).cpu()
    params["output_layernorm.weight"] = (model["model.norm.weight"]).cpu()
    params["lm_head.weight"] = (model["lm_head.weight"]).cpu()
    for i in range(num_layers):
        params[f"layers.{i}.ln_attn.weight"] = model[f"model.layers.{i}.input_layernorm.weight"].cpu()

        params[f"layers.{i}.attn.project_q.weight"] = (model[f"model.layers.{i}.self_attn.q_proj.weight"].transpose(0, 1)).cpu()
        params[f"layers.{i}.attn.project_k.weight"] = (model[f"model.layers.{i}.self_attn.k_proj.weight"].transpose(0, 1)).cpu()
        params[f"layers.{i}.attn.project_v.weight"] = (model[f"model.layers.{i}.self_attn.v_proj.weight"].transpose(0, 1)).cpu()

        params[f"layers.{i}.attn.attn_out.weight"] = (model[f"model.layers.{i}.self_attn.o_proj.weight"].transpose(0, 1)).cpu()

        params[f"layers.{i}.ln_ff.weight"] = model[f"model.layers.{i}.post_attention_layernorm.weight"].cpu()

        params[f"layers.{i}.ff.w_in.weight"] = (model[f"model.layers.{i}.mlp.gate_proj.weight"].transpose(0, 1)).cpu()
        params[f"layers.{i}.ff.w_gated.weight"] = (model[f"model.layers.{i}.mlp.up_proj.weight"].transpose(0, 1)).cpu()
        params[f"layers.{i}.ff.w_out.weight"] = (model[f"model.layers.{i}.mlp.down_proj.weight"].transpose(0, 1)).cpu()

    os.makedirs(f"/data/checkpoints/ultrallama/llama-2-70b/step_{step}_libcpm", exist_ok=True)
    model_name = f"/data/checkpoints/ultrallama/llama-2-70b/step_{step}_libcpm/model.ckpt"
    fout = open(model_name, "wb")
    fout.write( struct.pack("I", len(params)) )
    for name, value in params.items():
        print(f"Write ${name}")
        write_parameter(fout, name, value)
    fout.close()

    base_dir = "/mnt/data/user/tc_agi/user/zhaoweilin/libcpm-llama-2-70b"
    for n in ["tokenizer_config.json", "special_tokens_map.json", "tokenizer.model"]:
        if os.path.exists(os.path.join(base_dir, n)):
            shutil.copy(os.path.join(base_dir, n), os.path.join(f"/data/checkpoints/ultrallama/llama-2-70b/step_{step}_libcpm", n))


if __name__ == '__main__':
    # convert_model("llama-2-7b-hf", 32)
    # convert_model("llama-2-13b-hf", 40)
    convert_model(5800, 80)
