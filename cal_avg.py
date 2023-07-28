import json
import numpy as np
import os 

def get_num(name):
    name = name.split("-")[-1]
    dname = f"/data/eval_dataset/hendrycks_test-{name}"
    test_dname = os.path.join(dname, "test")
    if os.path.exists(test_dname):
        data = json.load(open(os.path.join(test_dname, "dataset_info.json")))
        return data["splits"]["test"]["num_examples"]
    else:
        print(test_dname)



data = json.load(open("./result_falcon_mmlu.json"))

print(len(data["results"]))
total = 0
acc_sum = 0
acc_norm_sum = 0
accs = []
acc_norms = []
for k in data["results"]:
    acc = data["results"][k]["acc"]
    accs.append(acc)
    acc_norm = data["results"][k]["acc_norm"]
    acc_norms.append(acc_norm)
    num = get_num(k)
    total += num
    acc_sum += acc * num
    acc_norm_sum += acc_norm * num

assert len(accs) == len(acc_norms) == 57

print("weighted average")
print("acc:", acc_sum/total)
print("acc norm:", acc_norm_sum/total)

print("unweighted average")
print("acc:", np.mean(accs))
print("acc norm:", np.mean(acc_norms))

