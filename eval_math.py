import json
import os
import transformers 
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import argparse
from datasets import load_dataset
import pandas as pd
import jsonlines
import re
import sys
from fraction import Fraction
from transformers import StoppingCriteria

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mine")
parser.add_argument("--alpha", type=float, default=0.0)
parser.add_argument("--cuda", type=int, default=2)
parser.add_argument("--prompt", type=str, default="What's your name?")
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--loadint8", type=int, default=2)
args = parser.parse_args()
device = f"cuda:{args.cuda}"

if args.model == "llama2-7b-chat":
    model_path = "/path/Llama-2-7b-chat-hf"
elif args.model == "llama2-7b":
    model_path = "/path/Llama-2-7b-hf"
elif args.model == "wizardmath-7b":
    model_path = "/path/WizardMath-7B-V1.0"

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence=[829, 29879, 29958]):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:, -len(self.eos_sequence):].tolist()
        return self.eos_sequence in last_ids

def extract_answer_number(completion):

    text = completion.split('The answer is: ')

    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        return None

tokenizer = AutoTokenizer.from_pretrained(model_path)

if args.loadint8 == 1:
    print("INT8")
    model = AutoModelForCausalLM.from_pretrained(model_path, load_in_8bit=True, device_map=device)
    generator = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer, 
        device_map="auto",
        eos_token_id=tokenizer.eos_token_id,
    )
elif args.loadint8 == 2:
    print("fp32")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    generator = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float32,
        tokenizer=tokenizer, 
        eos_token_id=tokenizer.eos_token_id,
        device=device,
    )


test_dataset = "./dataset/gsm8k_test_50.jsonl"

response_json = "./model_response/math_{model_name}.json".format(model_name=args.model)
f_json = open(response_json, mode='a')


if args.model == "wizardmath-7b":
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
else:
        problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request. Give the final number answer behind the string 'The answer is: '\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )


gsm8k_ins = []
gsm8k_answers = []
with open(test_dataset, "r+", encoding="utf8") as f:
    for idx, item in enumerate(jsonlines.Reader(f)):
        temp_instr = problem_prompt.format(instruction=item["question"])
        gsm8k_ins.append(temp_instr)
        temp_ans = item['answer'].split('#### ')[1]
        temp_ans = int(temp_ans.replace(',', ''))
        gsm8k_answers.append(temp_ans)

sum = 0

for idx in tqdm(range(50), desc='Processing'):

    sequences = generator(
        gsm8k_ins[idx],
        max_new_tokens=1024, 
        do_sample=True, 
        temperature=0.8, 
        top_k=50, 
        top_p=0.95,
        stopping_criteria=[EosListStoppingCriteria()])

    resp = sequences[0]['generated_text']
    response = {"resp": sequences[0]['generated_text']}
    json.dump(response, f_json, indent=4)

    answer_pred = extract_answer_number(resp)

    if answer_pred != None:
        result = float(answer_pred) == float(gsm8k_answers[idx])
    else:
        result = 0

    sum = sum + result

correctRate = sum / 50
json.dump(f"\n\n The Correct Rate on Math is {correctRate}", f_json, indent=4)
f_json.close()

with open("/path/math_fingerprint.json", "a") as f_total:
    f_total.write(json.dumps({
            "merge": args.model,
            "final rate": correctRate,
        }, ensure_ascii=False) + "\n")



