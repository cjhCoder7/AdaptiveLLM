#  0: Yi-Coder-1.5B-Chat, size: 1.5B
#  1: Qwen2.5-Coder-1.5B-Instruct, size: 1.5B
#  2: CodeLlama-7b-Instruct-hf, size: 7B
#  3: OpenCoder-8B-Instruct, size: 8B
#  4: starcoder2-15b-instruct-v0.1, size: 15B
#  5: deepseek-coder-v2-lite-instruct, size: 16B
#  6: Codestral-22B-v0.1, size: 22B
#  7: deepseek-coder-33b-instruct, size: 33B
#  8: Qwen2.5-Coder-32B-Instruct, size: 32B

import os
import json
import numpy as np

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

model_prices = [
    0.14,
    0.14,
    0.42,
    # 0.51,
    0.72,
    0.72,
    0.95,
    1.26,
    1.26,
]

datasets = [
    "HumanEval",
    "CodeContests",
    "Leetcode",
]

file_paths = []
for dataset in datasets:
    for model in models:
        file_paths.append(f"Result/{model}-output-{dataset}.jsonl")

problem_data = {}

for file_path in file_paths:
    model_name = file_path.split("-output-")[0].split("/")[-1]

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                problem_id = record["id"]
                prompt = record["prompt"]
                pass_rate = record["pass_rate"]
                avg_tokens = np.mean(record["output_token_counts"])
                model_price_idx = models.index(model_name)
                model_price = model_prices[model_price_idx]

                if problem_id not in problem_data:
                    problem_data[problem_id] = []

                problem_data[problem_id].append(
                    {
                        "model": model_name,
                        "prompt": prompt,
                        "avg_tokens": avg_tokens,
                        "pass_rate": pass_rate,
                        "model_price": model_price,
                    }
                )
    else:
        print(f"Wrong filename: {file_path}")

final_results = []

for problem_id in problem_data:
    problem_info = problem_data[problem_id]
    first_entry = problem_info[0]
    model_scores = []

    problem_prompt = first_entry["prompt"]

    all_tokens = [data["avg_tokens"] for data in problem_info]
    max_tokens = max(all_tokens) if all_tokens else 1
    all_model_sizes = [data["model_price"] for data in problem_info]
    max_model_size = max(all_model_sizes) if all_model_sizes else 1

    for data in problem_info:
        model_name = data["model"]
        avg_tokens = data["avg_tokens"]
        pass_rate = data["pass_rate"]
        model_size = data["model_price"]

        if max_tokens > 0 and max_model_size > 0 and avg_tokens > 0:
            term1 = np.log(max_tokens * max_model_size) * pass_rate
            term2 = np.log(avg_tokens * model_size)
            score = term1 - term2

            model_scores.append(
                {
                    "name": model_name,
                    "score": score,
                }
            )

    sorted_scores = sorted(model_scores, key=lambda x: x["score"], reverse=True)

    model_ranks = []
    for model_entry in sorted_scores:
        model_full_name = model_entry["name"]
        if model_full_name in models:
            model_idx = models.index(model_full_name)
            model_ranks.append(model_idx)

    final_entry = {
        "id": problem_id,
        "prompt": problem_prompt,
        "scores": sorted_scores,
        "rank": model_ranks,
    }

    final_results.append(final_entry)

with open("combined_results.jsonl", "w", encoding="utf-8") as f:
    for entry in final_results:
        json.dump(entry, f, ensure_ascii=False)
        f.write("\n")

for idx, entry in enumerate(final_results[:5], start=1):
    print(f"Problem {idx} - ID: {entry['id']}")
    for rank, model_score in enumerate(entry["scores"], start=1):
        print(f"  {rank}. {model_score['name']}: {model_score['score']:.2f}")
    print()
