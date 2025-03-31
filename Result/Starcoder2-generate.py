import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import transformers
import torch

device_map = "balanced"
model_path = "./starcoder2-15b-instruct-v0.1"

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
            if prompt_data["id"] in processed_ids:
                continue

            prompt = prompt_data.get("prompt", "")
            if not prompt:
                continue

            messages = [
                {"role": "user", "content": prompt},
            ]
            prompt_text = pipeline.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            input_token_count = len(tokenizer.encode(prompt_text))

            responses = []
            output_token_counts = []
            total_counts = []

            num_return_sequences = 5
            for _ in range(num_return_sequences):
                result = pipeline(
                    prompt_text,
                    max_new_tokens=2024,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.95,
                    top_k=20,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

                generated_text = result[0]["generated_text"]
                response = generated_text[len(prompt_text) :]

                responses.append(response)
                output_token_count = len(tokenizer.encode(response))
                output_token_counts.append(output_token_count)
                total_counts.append(input_token_count + output_token_count)

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


input_file = "prompts_en_extra_is_freeform.jsonl"
output_file = (
    "starcoder2-15b-instruct-v0.1-output-HumanEval.jsonl"
)

process_and_generate(input_file, output_file)
