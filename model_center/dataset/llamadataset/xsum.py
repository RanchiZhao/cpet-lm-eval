import torch
import json
import random

class XSum_Dataset(torch.utils.data.Dataset):
    def __init__(self, path, split, tokenizer, max_length) -> None:
        super().__init__()
        self.split = split
        self.data = []

        for input, target in self.read_data(path, split):
            if split == 'train':
                self.make_input(tokenizer, input, target, max_length)
            else:
                self.data.append({
                    "inputs": input + " Title:",
                    "targets": target
                })

    # def shift_tokens_right(self, input_ids, pad_token_id: int=0, decoder_start_token_id: int=0):
    #     shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    #     shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    #     shifted_input_ids[..., 0] = decoder_start_token_id
    #     shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    #     return shifted_input_ids

    def make_input(self, tokenizer, inputs, targets, max_length):
        model_inputs = {}
        
        answer = tokenizer(" Title: " + targets)["input_ids"][1:]
        # pad = [869] # .
        question = tokenizer(inputs)["input_ids"]
        head = question[0]
        if len(answer) + len(question) <= max_length:
            input_length = len(answer) + len(question)
            model_inputs['input_ids'] = torch.LongTensor(question + answer + [869] * (max_length - input_length))
        else:
            input_length = max_length
            space_left_for_question = input_length - len(answer)
            model_inputs['input_ids'] = torch.LongTensor(question[:space_left_for_question] + answer)
        
        
        label = tokenizer(targets)["input_ids"][1:]
        
        left_space = input_length - len(label) - 1
        right_space = max_length - input_length + 1
        label = [-100] * left_space + label + [tokenizer.eos_token_id] + [-100] * (right_space - 1) 
        model_inputs['targets'] = torch.LongTensor(label)
        
        input_length = torch.tensor(input_length, dtype=torch.int32)
        model_inputs['length'] = input_length
        
        self.data.append(model_inputs)
        

    def generate_input(self, question, context):
        return 

    def read_data(self, path, split):
        if split == 'dev':
            split = 'validation'
        if split == 'test': return
        path = f"{path}/{split}.json"
        with open(path, encoding='utf8') as f:
            f = json.load(f)
            for data in f:
                input = data['context']
                target = data['title']
                yield input, target
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.split == 'train':
            model_inputs = self.data[idx]
            for key, value in model_inputs.items():
                model_inputs[key] = value.cuda()
            return model_inputs
        else:
            return self.data[idx]