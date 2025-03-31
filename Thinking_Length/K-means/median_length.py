import json

file_names = [
    "DeepSeek-R1-Distill-Qwen-1.5B-output-total.jsonl",
    "DeepSeek-R1-Distill-Qwen-7B-output-total.jsonl",
    "DeepSeek-R1-Distill-Qwen-14B-output-total.jsonl",
    "DeepSeek-R1-Distill-Qwen-32B-output-total.jsonl",
]

output_file_names = [
    "DeepSeek-R1-Distill-Qwen-1.5B-output-total-median.jsonl",
    "DeepSeek-R1-Distill-Qwen-7B-output-total-median.jsonl",
    "DeepSeek-R1-Distill-Qwen-14B-output-total-median.jsonl",
    "DeepSeek-R1-Distill-Qwen-32B-output-total-median.jsonl",
]

for file_name, output_file_name in zip(file_names, output_file_names):
    with open(file_name, "r") as input_file, open(output_file_name, "w") as output_file:
        for line in input_file:
            item = json.loads(line)
            thinking_lengths = item["think_lengths"]
            thinking_lengths = [x for x in thinking_lengths if x is not None and x != 0]
            sorted_lengths = sorted(thinking_lengths)
            n = len(sorted_lengths)
            if n % 2 == 0:
                median = (sorted_lengths[n // 2 - 1] + sorted_lengths[n // 2]) / 2
            else:
                median = sorted_lengths[n // 2]
            item["median"] = median
            output_file.write(json.dumps(item) + "\n")
