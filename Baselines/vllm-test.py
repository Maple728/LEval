# -*- coding:utf-8 -*-
import os.path
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import openai
import numpy as np
import argparse
import tiktoken
import json

from tqdm import tqdm
from transformers import AutoTokenizer

from LEval_config import *


def num_tokens_from_string(string: str, tokenizer) -> int:
    if isinstance(tokenizer, str):
        encoding = tiktoken.encoding_for_model(tokenizer)
        num_tokens = len(encoding.encode(string))
    else:
        encoding = tokenizer(string, return_tensors="pt")
        num_tokens = len(encoding['input_ids'][0])
    return num_tokens


def process_one_record(d, sys_prompt, file_name):
    document = d['input']
    cnt = 0
    while num_tokens_from_string(document, tokenizer) > max_length:
        if "code" not in file_name:
            document = " ".join(document.split(" ")[:max_length - cnt])  # chunk the input len from right
        else:
            document = " ".join(document.split(" ")[cnt - max_length:])  # chunk the input len from left
        cnt += 250

    # print('document len', num_tokens_from_string(document, tokenizer))

    instructions = d['instructions']
    outputs = d['outputs']
    i = 0

    result_list = []
    for inst, out in zip(instructions, outputs):
        messages = [{"role": "system", "content": sys_prompt}]
        save_d = {}
        save_d['query'] = inst
        save_d['gt'] = out
        if "gsm" in file_name or "codeU" in file_name:
            messages.append({"role": "user", "content": document + "\n\n" + inst})
            save_d['prompt'] = sys_prompt + inst

        elif args.metric == "exam_eval":
            context = "Document is as follows. {} Question: {} \nPlease directly give answer without any additional output or explanation\n Answer: "
            messages.append({"role": "user", "content": context.format(document, inst)})
            save_d['prompt'] = sys_prompt + context
        else:
            context = "Document is as follows. {} Instruction: {} " + f"The suggested output length is around {len(out.split())} words. Output: "
            messages.append({"role": "user", "content": context.format(document, inst)})
            save_d['prompt'] = sys_prompt + context

        for _ in range(10):
            try:
                # if start_idx == 0:
                #     print(messages[1]["content"])
                #     print("--------------------------- end of example input ------------------")
                #     input("Press Enter to confirm this is the correct input for the api call ...")
                #     start_idx += 1
                response = openai.chat.completions.create(
                    model=openai_model,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=0.0,
                )  # get response

                ret = response.choices[0].message.content
                ret = ret.strip()  # get the paraphrased answer

                save_d[f'{openai_model}_pred'] = ret
                save_d['evaluation'] = d['evaluation']

                # test the factuality in scientific fiction
                if "sci_fi" in file_name:
                    text_inputs = inst.replace("based on the world described in the document.",
                                               "based on the real-world knowledge and facts up until your last training") + "\nPlease directly give answer without any additional output or explanation \nAnswer:"
                    messages.append({"role": "user", "content": text_inputs})
                    response = openai.chat.completions.create(
                        model=openai_model,
                        messages=messages,
                        max_tokens=max_new_tokens,
                        temperature=0.0,
                    )  # get response
                    ret = response.choices[0].message.content
                    ret = ret.strip()  # get the paraphrased answer
                    save_d[f'{openai_model}_pred'] += f" [fact: {ret}]"

                # print("----------------- [output] vs [ground truth] -----------------")
                # print('[output]:', save_d[f'{openai_model}_pred'], "\n\n", '[ground truth]:', save_d['gt'])
                result_list.append(save_d)
                break

            except Exception as e:  # add some logit here for retry
                if isinstance(e, KeyboardInterrupt):
                    raise e
                print(i, e)

                time.sleep(0.8)
    return result_list


def main():

    pool = ThreadPoolExecutor(max_workers=16)
    for file_name in key_data_pairs:
        sys_prompt = get_sys_prompt(args, file_name)
        fw = open(f'{file_name}', "w")
        data = key_data_pairs[file_name]

        data_len = len(data)
        iterator = pool.map(
            process_one_record,
            data,
            [sys_prompt] * data_len,
            [file_name] * data_len,
        )
        print('processing file:', file_name)
        for save_d_list in tqdm(iterator, total=data_len):
            for save_d in save_d_list:
                fw.write(json.dumps(save_d) + '\n')
        # for d in tqdm(data):
        #     save_d = process_one_record(d, sys_prompt, file_name)
        #     fw.write(json.dumps(save_d) + '\n')

        fw.close()
        # break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--openai_url', help='openai url')
    parser.add_argument('--openai_api_key', help='openai api key')
    parser.add_argument('--model', help='openai api model')
    parser.add_argument('--tokenizer', help='tokenizer path')

    parser.add_argument('--metric',
                        choices=["llm_turbo_eval", "llm_gpt4_eval", "exam_eval", "ngram_eval", "human_eval"],
                        required=True, help='metric name from ["turbo_eval","gpt4_eval","auto_eval", ...]')

    parser.add_argument('--max_length', default="16k", help='max length of the input, e.g., 2k, 16k')
    # if none, we will load from huggingface
    parser.add_argument('--task_path', type=str, default=None,
                        help='set this if you want test a specific task , example: LEval-data/Closed-ended-tasks/coursera.jsonl or LEval-data/Closed-ended-tasks/ ')
    parser.add_argument('--task_name', type=str, default=None,
                        help='optional, if not set, we will test all. set this if you want test a specific task from huggingface, example: coursera, tpo')
    parser.add_argument('--mc_tasks', action='store_true')
    args = parser.parse_args()

    openai.api_key = args.openai_api_key
    openai.base_url = args.openai_url

    key_data_pairs = {}

    max_length = k_to_number(args.max_length) - max_new_tokens
    openai_model = args.model
    data_save_path = f"Predictions/{args.metric}/{openai_model}"

    if args.tokenizer is None:
        tokenizer = 'gpt-3.5-turbo'
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # input(f"Your prediction file will be saved to: {data_save_path}  , press enter to confirm...")
    print(f"Your prediction file will be saved to: {data_save_path}")
    build_key_data_pairs(args, key_data_pairs, data_save_path)
    sys.exit(main())
