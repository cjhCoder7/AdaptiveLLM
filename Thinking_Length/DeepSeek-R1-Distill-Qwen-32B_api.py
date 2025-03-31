from openai import OpenAI, APIError, APIConnectionError
import json
import time

client = OpenAI(
    api_key="API_KEY",
    base_url="https://api.siliconflow.com/v1/",
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


def call_api(prompt, num_responses):
    user_prompt = prompt + "\nYou should initiate the response with <think>"
    messages = [{"role": "user", "content": user_prompt}]
    think_lengths = []

    for _ in range(num_responses):
        max_retries = 10
        retry_count = 0
        success = False
        think_length = 0

        while retry_count < max_retries and not success:
            try:
                completion = client.chat.completions.create(
                    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                    messages=messages,
                    stream=True,
                    max_tokens=16384,
                )

                reasoning_content = ""
                for chunk in completion:
                    reasoning_chunk = chunk.choices[0].delta.reasoning_content or ""
                    answer_chunk = chunk.choices[0].delta.content or ""

                    if reasoning_chunk:
                        print(reasoning_chunk, end="")
                        reasoning_content += reasoning_chunk
                    if answer_chunk:
                        print(answer_chunk, end="")

                think_length = len(reasoning_content)
                success = True

            except (APIError, APIConnectionError) as e:
                print(f"\nError: {str(e)}")
                retry_count += 1

                wait_time = 2 ** (retry_count - 1)
                print(f"Retrying ({retry_count}/{max_retries}) in {wait_time}s...")
                time.sleep(wait_time)

            except Exception as e:
                print(f"\nCritical error: {str(e)}")
                break

        if success:
            think_lengths.append(think_length)
        else:
            think_lengths.append(None)
            print("Failed after maximum retries")

    return think_lengths


def process_and_generate(input_file, output_file):
    prompts = read_jsonl(input_file)

    processed_ids = get_processed_ids(output_file)

    count = 0

    with open(output_file, "a", encoding="utf-8") as out_file:
        for prompt_data in prompts:
            if count <= 13:
                count += 1
                continue

            count += 1
            print(f"Processed {count} prompts.")
            print("=============================================")

            if prompt_data["id"] in processed_ids:
                continue

            prompt = prompt_data.get("prompt", "")
            if not prompt:
                continue

            num_return_sequences = 10
            think_lengths = call_api(prompt, num_return_sequences)

            prompt_data["think_lengths"] = think_lengths

            out_file.write(json.dumps(prompt_data, ensure_ascii=False) + "\n")


input_file = "prompts_en_extra_is_freeform.jsonl"
output_file = (
    "DeepSeek-R1-Distill-Qwen-32B-output-HumanEval.jsonl"
)

process_and_generate(input_file, output_file)
