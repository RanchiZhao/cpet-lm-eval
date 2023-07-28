import argparse
import json
import logging
import os
import numpy as np
import random
import lm_eval.models
import lm_eval.tasks
import lm_eval.base
from lm_eval.utils import positional_deprecated, run_task_tests

from lm_eval import tasks, evaluator, utils

logging.getLogger("openai").setLevel(logging.WARNING)

def load_model(args):
    random.seed(1234)
    np.random.seed(1234)


    if isinstance(args.model, str):
        if args.model_args is None:
            args.model_args = ""
        lm = lm_eval.models.get_model(args.model).create_from_arg_string(
            args.model_args, {"batch_size": args.batch_size, "max_batch_size": args.max_batch_size, "device": args.device}
        )
    else:
        assert isinstance(args.model, lm_eval.base.LM)
        lm = args.model

    # if not no_cache:
    #     lm = lm_eval.base.CachingLM(
    #         lm,
    #         "lm_cache/"
    #         + (args.model if isinstance(args.model, str) else args.model.model.config._name_or_path)
    #         + "_"
    #         + args.model_args.replace("=", "-").replace(",", "_").replace("/", "-")
    #         + ".db",
    #     )
    

    return lm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument("--tasks", default=None, choices=utils.MultiChoice(tasks.ALL_TASKS))
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default=None)
    parser.add_argument("--max_batch_size", type=int, default=None,
                        help="Maximal batch size to try with --batch_size auto")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--limit", type=float, default=None,
                        help="Limit the number of examples per task. "
                             "If <1, limit is a percentage of the total number of examples.")
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    assert not args.provide_description  # not implemented

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    # if args.tasks is None:
    #     task_names = tasks.ALL_TASKS
    # else:
    #     task_names = utils.pattern_match(args.tasks.split(","), tasks.ALL_TASKS)
    
    # assert task_names != [], "No tasks specified"

    # task_dict = lm_eval.tasks.get_task_dict(task_names)

    # print(f"Selected Tasks: {task_names}")
    # print(task_dict)

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    lm = load_model(args)

    for tasks_str, num_fewshot in zip(["arc_challenge_sample", "hellaswag_sample", "truthfulqa_mc_sample", "hendrycksTest_sample-*"], [25, 10, 0, 5]):
        if os.path.exists(os.path.join(args.output_path, f"eval_{tasks_str}.json")):
            continue
        task_names = utils.pattern_match(tasks_str.split(","), tasks.ALL_TASKS)
    
        assert task_names != [], "No tasks specified"

        task_dict = lm_eval.tasks.get_task_dict(task_names)
        print(f"Selected Tasks: {task_names}")
        print(f"Num fewshot: {num_fewshot}")
        print(task_dict)
        
        results = evaluator.evaluate(
            lm=lm,
            task_dict=task_dict,
            num_fewshot=num_fewshot,
            limit=args.limit,
            description_dict=description_dict,
            decontamination_ngrams_path=args.decontamination_ngrams_path,
            write_out=args.write_out,
            output_base_path=args.output_base_path,
        )

        results["config"] = {
            "model": (args.model if isinstance(args.model, str) else args.model.model.config._name_or_path),
            "model_args": args.model_args,
            "num_fewshot": num_fewshot,
        }

        dumped = json.dumps(results, indent=2)
        def is_mmlu(data):
            for k in data["results"]:
                if "hendrycksTest" in k:
                    return True
            return False
        if is_mmlu(results):
            accs = []
            for k in results["results"]:
                accs.append(results["results"][k]["acc_norm"])
            print("mmlu acc_norm unweighted average:", np.mean(accs))
            results["results"]["unweighted_average_acc_norm"] = np.mean(accs)
            dumped = json.dumps(results)
        else:
            print(dumped)

        # if args.output_path:
        with open(os.path.join(args.output_path, f"eval_{tasks_str}.json"), "w") as f:
            f.write(dumped)

        # batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))
        # print(
        #     f"{args.model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
        #     f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
        # )
        # if not is_mmlu(results):
        #     print(evaluator.make_table(results))


if __name__ == "__main__":
    main()
