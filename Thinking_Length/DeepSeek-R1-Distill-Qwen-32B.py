import json
import re
from transformers import pipeline
import transformers
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device_map = "auto"
model_path = "./DeepSeek-R1-Distill-Qwen-32B"
string = "\nYou should initiate the response with <think>\n"

pipeline = transformers.pipeline(
    model=model_path,
    task="text-generation",
    torch_dtype=torch.bfloat16,
    device_map=device_map,
    trust_remote_code=True,
)

tokenizer = pipeline.tokenizer


def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def get_processed_ids(output_file):
    processed_ids = set()
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                if "id" in record:
                    processed_ids.add(record["id"])
    except FileNotFoundError:
        pass
    return processed_ids


def process_and_generate(input_file, output_file):
    prompts = read_jsonl(input_file)

    processed_ids = get_processed_ids(output_file)

    count = 0

    with open(output_file, "a", encoding="utf-8") as out_file:
        for prompt_data in prompts:
            count += 1
            print(f"Processed {count} prompts.")
            print("=============================================")

            if prompt_data["id"] in processed_ids:
                continue

            prompt = prompt_data.get("prompt", "")
            if not prompt:
                continue

            messages = [
                {"role": "user", "content": prompt + string},
            ]
            prompt_text = pipeline.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            think_lengths = []

            num_return_sequences = 10
            for i in range(num_return_sequences):
                result = pipeline(prompt_text, max_new_tokens=30000)

                generated_text = result[0]["generated_text"]
                response = generated_text[len(prompt_text) :]

                print(response)
                print("-----------------------------------")

                think_blocks = re.findall(r"<think>(.*?)</think>", response, re.DOTALL)
                if think_blocks:
                    total_length = sum(len(block) for block in think_blocks)
                    think_lengths.append(total_length)
                    print(
                        f"Request {i+1}: Think block length = {total_length}, {len(think_blocks)}\n\n"
                    )
                else:
                    think_lengths.append(0)
                    print(f"Request {i+1}: No think block found\n\n")

            prompt_data["think_lengths"] = think_lengths

            out_file.write(json.dumps(prompt_data, ensure_ascii=False) + "\n")


input_file = "prompts_en_extra_is_freeform.jsonl"
output_file = (
    "DeepSeek-R1-Distill-Qwen-32B-output-HumanEval.jsonl"
)

process_and_generate(input_file, output_file)
