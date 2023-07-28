import json
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer, GenerationConfig
from tqdm import tqdm
import sys, os
import torch

# model_name = "/data/ultrallama/ultrallama-13b-modelcenter-fulldata-step2000_hf"
# model_name = "cyl/awsome-llama"
model_name = "/data/ultrallama/ultrallama-13b_reasoning_step3400_hf"
if __name__ == "__main__":
    model_name = sys.argv[-1]

    eval_set = json.load(open("./alpaca_eval.json", encoding="utf-8"))
    print(len(eval_set))

    tokenizer = LlamaTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto",
            torch_dtype=torch.float16)

    # model = model.cuda()
    # default temperature=0.7, do_sample=True, top_p=1.0
    for temp in [0.7]:
        for do_sample in [True]:
            for top_p in [1.0]:
                generation_config = GenerationConfig(max_new_tokens=2000, do_sample=do_sample, num_beams=1, early_stopping=True, temperature=temp, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, top_p=top_p)
                system_prompt = "Please give helpful, detailed, and polite answers to the user's questions."
                system_prompt = "Please give helpful, very detailed, and polite answers to the user's questions."
                prompts = []
                if "13b" in model_name:
                    for example in eval_set:
                        q_input = "User: " + system_prompt + tokenizer.eos_token + "\nUser: " + example["instruction"] + tokenizer.eos_token + "\nAssistant: "
                        prompts.append(q_input)
                elif "65b" in model_name:
                    # system_prompt = "Please give helpful, detailed, and polite answers to the user's question: "
                    for example in eval_set:
                        q_input = "User: " + system_prompt + "\nUser: " + example["instruction"] + "\nAssistant: "
                        prompts.append(q_input)
                else:
                    raise ValueError(f"not recognized model_name: {model_name}")
                
                print("example prompt:\n", prompts[0])
                result = []
                idx = 0
                bs = 1
                import math
                iters = math.ceil(len(prompts)/bs)
                for i in tqdm(range(iters)):
                    # print(q_input)
                    inputs = tokenizer(prompts[i*bs:(i+1)*bs], return_tensors="pt", padding="longest")
                    inference_results = model.generate(inputs["input_ids"].cuda(), generation_config=generation_config)
                    input_length = inputs["input_ids"].size(-1)
                    # input_length = [len(seq) for seq in inputs["input_ids"]]
                    outputs = tokenizer.batch_decode(inference_results[:, input_length:], skip_special_tokens=True)
                    for i, output in enumerate(outputs):
                        # print(output)
                        example = eval_set[idx]
                        example["output"] = output
                        example["generator"] = model_name
                        result.append(example)
                        idx += 1
                    # break

                with open(os.path.join(model_name, f"result_alpaca_eval_temperature_{temp}_sample_{do_sample}_topp_{top_p}.json"), "w")as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print("result saved at ", os.path.join(model_name, f"result_alpaca_eval_temperature_{temp}_sample_{do_sample}_topp_{top_p}.json"))
