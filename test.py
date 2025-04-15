# -*- coding: utf-8 -*-
"""
@author: Sebastian Riedel <sriedel@suse.com>
"""
import argparse
from datasets import load_dataset
from datetime import datetime, timedelta
import numpy as np
from timeit import default_timer as timer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# model_path = "/tmp/Meta-Llama-3-8B-Instruct-Cavil-hf"
model_path = "../Meta-Llama-3-8B-Instruct"
# model_path = "../Llama-2-7b-chat-hf"
# model_path = "../Mistral-7B-Instruct-v0.2"
# model_path = "../Phi-3-mini-4k-instruct"

data_path = "legaldb-ml-data-small.jsonl"

# We were having trouble with Apple M1, better to use CPU for now
device = "cpu"
torch_dtype = torch.float32
if torch.cuda.is_available():
    device = "cuda"
    torch_dtype = torch.bfloat16


system_prompt = """
You are a helpful lawyer. Analyze the code or documentation snippet enclosed
in "[CODE]" and "[/CODE]" tokens to determine if it contains legal text that
was written with the intention of describing how the code should be used.
Answer only with "yes" or "no".

User:
[CODE]// SPDX-License-Identifier: MIT[/CODE]
Assistant:
yes

User:
[CODE]// Released under BSD-2-clause license[/CODE]
Assistant:
yes

User:
[CODE]# Released under BSD-3-clause license[/CODE]
Assistant:
yes

User:
[CODE]Hello World[/CODE]
Assistant:
no

User:
[CODE]Foo Bar Baz[/CODE]
Assistant:
no

User:
[CODE]GPL License Version 2.0[/CODE]
Assistant:
yes

User:
[CODE]// Copyright 2024
//Licensed as BSD-3-clause
[/CODE]
Assistant:
yes

User:
[CODE]my $foo = 23;[/CODE]
Assistant:
no

User:
[CODE]
# SPDX-License-Identifier: MIT
my $foo = 23;
[/CODE]
Assistant:
yes

User:
[CODE]if (license === true) {[/CODE]
Assistant:
no

Analyze the following code or documentation snippet. Answer only with "yes" or "no".
"""


def get_args():
    parser = argparse.ArgumentParser("Test LegalDB models")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=data_path,
        help="path to input file",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=model_path,
        help="path to model",
    )
    parser.add_argument(
        "-q",
        "--quant",
        type=str,
        choices=["4", "8"],
        default=None,
        help="quantize model",
    )
    return parser.parse_args()


args = get_args()
load_in_8bit = False
load_in_4bit = False
if args.quant == "8":
    load_in_8bit = True
elif args.quant == "4":
    load_in_4bit = True
model = AutoModelForCausalLM.from_pretrained(
    args.model, device_map=device, torch_dtype=torch_dtype, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit
)
tokenizer = AutoTokenizer.from_pretrained(args.model)
eos_token_id = tokenizer.encode("\n")


def get_prompt(snippet):
    return f"{system_prompt}\nUser:\n[CODE]{snippet}[/CODE]\nAssistant:\n"


def get_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    outputs = model.generate(
        inputs=inputs,
        num_return_sequences=1,
        max_new_tokens=1,
        top_p=None,
        temperature=None,
        do_sample=False,
        eos_token_id=eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        output_scores=True,
        return_dict_in_generate=True,
    )

    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )
    input_length = inputs.shape[1]
    generated_tokens = outputs.sequences[:, input_length:]
    token = generated_tokens[0][0]
    score = transition_scores[0][0].cpu()

    return {
        "token": f"{token}",
        "text": tokenizer.decode(token),
        "timestamp": datetime.timestamp(datetime.now()),
        "score": f"{score.numpy():.4f}",
        "confidence": f"{np.exp(score.numpy()):.2%}",
    }


dataset = load_dataset("json", data_files=args.input)
correct = 0
start = timer()
for data_point in dataset["train"]:
    is_legal_text = data_point["is_legal_text"]
    snippet = data_point["snippet"][:2048]
    response = get_response(get_prompt(snippet))
    print(f"{is_legal_text}: " + str(response))
    result = response["text"].lower()
    if is_legal_text and result == "yes":
        correct += 1
    elif not is_legal_text and result == "no":
        correct += 1
end = timer()

print(f"Accuracy: {correct / len(dataset['train'])}")
delta = timedelta(seconds=end - start)
print(f"Time: {delta}")
