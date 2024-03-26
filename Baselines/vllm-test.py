# -*- coding:utf-8 -*-
import os.path
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import openai
import numpy as np
import argparse
import tiktoken
import json

from tqdm import tqdm
from transformers import AutoTokenizer

from LEval_config import max_new_tokens, get_sys_prompt, build_key_data_pairs, k_to_number


def num_tokens_from_string(string: str, tokenizer) -> int:
    if isinstance(tokenizer, str):
        encoding = tiktoken.encoding_for_model(tokenizer)
        num_tokens = len(encoding.encode(string))
    else:
        encoding = tokenizer(string, return_tensors="pt")
        num_tokens = len(encoding['input_ids'][0])
    return num_tokens


def process_one_record(record, inst, output, meta_config):
    sys_prompt = meta_config['sys_prompt']
    file_name = meta_config['file_name']
    tokenizer = meta_config['tokenizer']
    max_length = meta_config['max_length']
    model = meta_config['model']
    metric = meta_config['metric']

    document = record['input']
    cnt = 0
    while num_tokens_from_string(document, tokenizer) > max_length:
        if "code" not in file_name:
            document = " ".join(document.split(" ")[:max_length - cnt])  # chunk the input len from right
        else:
            document = " ".join(document.split(" ")[cnt - max_length:])  # chunk the input len from left
        cnt += 250

    # print('document len', num_tokens_from_string(document, tokenizer))

    messages = [{"role": "system", "content": sys_prompt}]
    save_d = {}
    save_d['query'] = inst
    save_d['gt'] = output
    if "gsm" in file_name or "codeU" in file_name:
        messages.append({"role": "user", "content": document + "\n\n" + inst})
        save_d['prompt'] = sys_prompt + inst

    elif metric == "exam_eval":
        context = "Document is as follows. {} Question: {} \nPlease directly give answer without any additional output or explanation\n Answer: "
        messages.append({"role": "user", "content": context.format(document, inst)})
        save_d['prompt'] = sys_prompt + context
    else:
        context = "Document is as follows. {} Instruction: {} " + f"The suggested output length is around {len(out.split())} words. Output: "
        messages.append({"role": "user", "content": context.format(document, inst)})
        save_d['prompt'] = sys_prompt + context

    for i in range(10):
        try:
            # if start_idx == 0:
            #     print(messages[1]["content"])
            #     print("--------------------------- end of example input ------------------")
            #     input("Press Enter to confirm this is the correct input for the api call ...")
            #     start_idx += 1
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=0.0,
            )  # get response

            ret = response.choices[0].message.content
            ret = ret.strip()  # get the paraphrased answer

            save_d[f'{model}_pred'] = ret
            save_d['evaluation'] = record['evaluation']

            # test the factuality in scientific fiction
            if "sci_fi" in file_name:
                text_inputs = inst.replace("based on the world described in the document.",
                                           "based on the real-world knowledge and facts up until your last training") + "\nPlease directly give answer without any additional output or explanation \nAnswer:"
                messages.append({"role": "user", "content": text_inputs})
                response = openai.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=0.0,
                )  # get response
                ret = response.choices[0].message.content
                ret = ret.strip()  # get the paraphrased answer
                save_d[f'{model}_pred'] += f" [fact: {ret}]"

            # print("----------------- [output] vs [ground truth] -----------------")
            # print('[output]:', save_d[f'{openai_model}_pred'], "\n\n", '[ground truth]:', save_d['gt'])
            return save_d

        except Exception as e:  # add some logit here for retry
            if isinstance(e, KeyboardInterrupt):
                raise e
            print('invoke', i, e)
            time.sleep(0.8)


def main(key_data_pairs, model, max_length, tokenizer, metric, num_workers):
    pool = ThreadPoolExecutor(max_workers=num_workers)
    for file_name in key_data_pairs:
        sys_prompt = get_sys_prompt(args, file_name)
        fw = open(f'{file_name}', "w")
        data = key_data_pairs[file_name]

        record_list = []
        inst_list = []
        output_list = []
        for d in data:
            instructions = d['instructions']
            outputs = d['outputs']
            for inst, out in zip(instructions, outputs):
                record_list.append(d)
                inst_list.append(inst)
                output_list.append(out)

        total_len = len(inst_list)
        iterator = pool.map(
            process_one_record,
            record_list,
            inst_list,
            output_list,
            [{
                'file_name': file_name,
                'sys_prompt': sys_prompt,
                'model': model,
                'max_length': max_length,
                'tokenizer': tokenizer,
                'metric': metric,

            }] * total_len,
        )
        print('processing file:', file_name)
        for save_d in tqdm(iterator, total=total_len):
            fw.write(json.dumps(save_d) + '\n')
        fw.close()


def eval_by_vllm(
        openai_url: str,
        model: str,
        task_args,
        openai_api_key: str = None,
        output_path: str = None,
        tokenizer=None,
        num_workers: int = 16,
):
    openai.base_url = openai_url
    openai.api_key = openai_api_key
    key_data_pairs = {}

    max_length = k_to_number(task_args.max_length) - max_new_tokens

    if output_path is None:
        output_path = f"Predictions/{task_args.metric}/{model}"

    if tokenizer is None:
        tokenizer = 'gpt-3.5-turbo'
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

    print(f"Your prediction file will be saved to: {output_path}")
    build_key_data_pairs(args, key_data_pairs, output_path)

    main(key_data_pairs, model, max_length, tokenizer, metric=task_args.metric, num_workers=num_workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--openai_url', help='openai url')
    parser.add_argument('--openai_api_key', help='openai api key')
    parser.add_argument('--model', help='openai api model')
    parser.add_argument('--tokenizer', help='tokenizer path')
    parser.add_argument('--output', help='Prediction output path')

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

    eval_by_vllm(
        openai_url=args.openai_url,
        openai_api_key=args.openai_api_key,
        model=args.model,
        output_path=args.output,
        tokenizer=args.tokenizer,
        task_args=args,
    )
