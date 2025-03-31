import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 3, 4"
device_map = "auto"
model_path = "./Qwen2.5-Coder-32B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map=device_map, torch_dtype=torch.bfloat16
)


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
            if prompt_data["id"] in processed_ids:
                continue

            prompt = prompt_data.get("prompt", "")
            if not prompt:
                continue

            messages = [
                {
                    "role": "system",
                    "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                },
                {"role": "user", "content": prompt},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            input_token_count = model_inputs.input_ids.shape[1]

            responses = []
            output_token_counts = []
            total_counts = []

            num_return_sequences = 5
            for _ in range(num_return_sequences):
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=2024,
                    # do_sample=True,
                    # temperature=0.3,
                    # top_p=0.95,
                    # top_k=20,
                )

                generated_ids = [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(
                        model_inputs.input_ids, generated_ids
                    )
                ]

                response = tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]
                responses.append(response)
                ouput_token_count = len(tokenizer.encode(response))
                output_token_counts.append(ouput_token_count)
                total_counts.append(input_token_count + ouput_token_count)

            prompt_data["responses"] = responses
            prompt_data["input_token_count"] = input_token_count
            prompt_data["output_token_counts"] = output_token_counts
            prompt_data["total_token_counts"] = total_counts

            out_file.write(json.dumps(prompt_data, ensure_ascii=False) + "\n")

            count += 1
            print(f"Processed {count} prompts.")
            print(f"total_token_counts: {total_counts}")
            print("The first response is：")
            print(responses[0])


input_file = "prompts_python_en_test.jsonl"  
output_file = (
    "Qwen2.5-Coder-32B-Instruct-output-CodeContests.jsonl"  
)

process_and_generate(input_file, output_file)
