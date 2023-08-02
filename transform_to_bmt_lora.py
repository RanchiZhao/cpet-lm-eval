import torch, os
import json
from collections import OrderedDict
from tqdm import tqdm
import os

def transform_to_hf(bmt_model):
    model_hf = OrderedDict()
    
    for k, v in bmt_model.items():
        if 'lora' in k:
            model_hf[k] = v
    return model_hf  


if __name__ == "__main__":
    import sys
    import shutil
    in_path = sys.argv[-1]
    
    out_path = os.path.join(, "lora_9.pt")
    # os.makedirs(out_path, exist_ok=True)
    # if not os.path.exists(out_path):
    print("transforming...")
    hf_state_dict = transform_to_hf(torch.load(in_path))
    print(hf_state_dict.keys())
    print("done")
    torch.save(hf_state_dict, out_path)
        
    print("saved")
    print(list(os.path.dirname(in_path)))




