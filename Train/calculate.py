import json

models = [
    "Yi-Coder-1.5B-Chat",
    "Qwen2.5-Coder-1.5B-Instruct",
    "CodeLlama-7b-Instruct-hf",
    # "OpenCoder-8B-Instruct",
    "starcoder2-15b-instruct-v0.1",
    "deepseek-coder-v2-lite-instruct",
    "Codestral-22B-v0.1",
    "deepseek-coder-33b-instruct",
    "Qwen2.5-Coder-32B-Instruct",
]

#  0: Yi-Coder-1.5B-Chat, size: 1.5B
#  1: Qwen2.5-Coder-1.5B-Instruct, size: 1.5B
#  2: CodeLlama-7b-Instruct-hf, size: 7B
#  3: OpenCoder-8B-Instruct, size: 8B
#  4: starcoder2-15b-instruct-v0.1, size: 15B
#  5: deepseek-coder-v2-lite-instruct, size: 16B
#  6: Codestral-22B-v0.1, size: 22B
#  7: deepseek-coder-33b-instruct, size: 33B
#  8: Qwen2.5-Coder-32B-Instruct, size: 32B

datasets = [
    "HumanEval",
    "CodeContests",
    "Leetcode",
]

model_prices = [
    0.14,
    0.14,
    0.42,
    # 0.42,
    0.72,
    0.72,
    0.95,
    1.26,
    1.26,
]

file_paths = []
for dataset in datasets:
    for model in models:
        file_paths.append(f"Result/{model}-output-{dataset}.jsonl")

file_name = "predictions_2.jsonl"
all_data = []
with open(file_name, "r") as f:
    for line in f:
        all_data.append(json.loads(line))


pass_1 = 0
pass_5 = 0
token_1 = 0
token_5 = 0
token_1_cost = 0
token_5_cost = 0
count = 0

for item in all_data:
    count += 1
    id = item["id"]
    predicted_rank = item["predicted_rank"]

    selected_model = models[predicted_rank]

    if id.startswith("HumanEval"):
        selected_dataset = "HumanEval"
    elif id.startswith("code_contests"):
        selected_dataset = "CodeContests"
    elif id.startswith("LeetCode"):
        selected_dataset = "Leetcode"

    for file_path in file_paths:
        if selected_model in file_path and selected_dataset in file_path:
            with open(file_path, "r") as f:
                for line in f:
                    data = json.loads(line)
                    if data["id"] == id:
                        # print(selected_model)
                        # print(selected_dataset)
                        # print(count)
                        # print("=" * 50)
                        model_index = models.index(selected_model)
                        model_price = model_prices[model_index]

                        if data["results"][0]:
                            pass_1 += 1
                        if any(data["results"]):
                            pass_5 += 1
                        token_1 += data["output_token_counts"][0]
                        token_1_cost += (
                            data["output_token_counts"][0] * model_price / 1000000
                        )

                        temp_token = 0
                        for i in range(5):
                            if data["output_token_counts"][i]:
                                temp_token += data["output_token_counts"][i]
                        token_5 += temp_token
                        token_5_cost += temp_token * model_price / 1000000
                        break

print("count:", count)
print(f"pass@1: {pass_1 / count}")
print(f"pass@5: {pass_5 / count}")
print(f"token@1: {token_1 / count}")
print(f"token@5: {token_5 / count}")
print(f"token@1 cost: {token_1_cost / count}")
print(f"token@5 cost: {token_5_cost / count}")

for model in models:
    pass_1 = 0
    pass_5 = 0
    token_1 = 0
    token_5 = 0
    token_1_cost = 0
    token_5_cost = 0
    model_index = models.index(model)
    model_price = model_prices[model_index]

    for item in all_data:
        id = item["id"]
        selected_dataset = ""
        if id.startswith("HumanEval"):
            selected_dataset = "HumanEval"
        elif id.startswith("code_contests"):
            selected_dataset = "CodeContests"
        elif id.startswith("LeetCode"):
            selected_dataset = "Leetcode"

        for file_path in file_paths:
            if model in file_path and selected_dataset in file_path:
                with open(file_path, "r") as f:
                    for line in f:
                        data = json.loads(line)
                        if data["id"] == id:
                            if data["results"][0]:
                                pass_1 += 1
                            if any(data["results"]):
                                pass_5 += 1
                            token_1 += data["output_token_counts"][0]
                            token_1_cost += (
                                data["output_token_counts"][0] * model_price / 1000000
                            )
                            temp_token = 0
                            for i in range(5):
                                if data["output_token_counts"][i]:
                                    temp_token += data["output_token_counts"][i]
                            token_5 += temp_token
                            token_5_cost += temp_token * model_price / 1000000
                            break
    print("=" * 50)
    print(model)
    print(f"pass@1: {pass_1 / count}")
    print(f"pass@5: {pass_5 / count}")
    print(f"token@1: {token_1 / count}")
    print(f"token@5: {token_5 / count}")
    print(f"token@1 cost: {token_1_cost / count}")
    print(f"token@5 cost: {token_5_cost / count}")
