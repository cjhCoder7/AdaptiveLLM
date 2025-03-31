import json
import random
from collections import defaultdict

data = []
with open("combined_results.jsonl", "r") as file:
    for line in file:
        data.append(json.loads(line))

grouped_data = defaultdict(list)
for item in data:
    if item["id"].startswith("HumanEval"):
        grouped_data["HumanEval"].append(item)
    elif item["id"].startswith("code_contests"):
        grouped_data["CodeContests"].append(item)
    elif item["id"].startswith("LeetCode"):
        grouped_data["LeetCode"].append(item)

train_ratio = 0.8
test_ratio = 0.2

train_data = []
test_data = []

for group in grouped_data.values():
    random.shuffle(group)
    train_size = int(len(group) * train_ratio)
    train_data.extend(group[:train_size])
    test_data.extend(group[train_size:])

with open("train_data.jsonl", "w") as train_file:
    for item in train_data:
        train_file.write(json.dumps(item) + "\n")

with open("test_data.jsonl", "w") as test_file:
    for item in test_data:
        test_file.write(json.dumps(item) + "\n")
