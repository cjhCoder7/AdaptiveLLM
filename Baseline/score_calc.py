import json

print("Baseline")
with open("CodeLlama_processed_results.jsonl", "r", encoding='utf-8') as f1, \
     open("GPT3.5_processed_results.jsonl", "r", encoding='utf-8') as f2, \
     open("GPT4o_processed_results.jsonl", "r", encoding='utf-8') as f3:
    
    score1 = 0
    score5 = 0
    cost1 = 0
    cost5 = 0
    token1 = 0
    token5 = 0
    cnt = 0

    for line in f1:
        entry = json.loads(line)
        pass_in_1 = entry.get("pass_in_1")
        pass_in_5 = entry.get("pass_in_5")
        score1 += 1 if pass_in_1 else 0
        score5 += 1 if pass_in_5 else 0
        token_counts = entry.get("output_token_counts", [])
        token1 += token_counts[0]
        token5 += sum(token_counts)
        cost1 += token_counts[0] * 0.42 / 1000000
        cost5 += sum(token_counts) * 0.42 / 1000000
        cnt += 1

    for line in f2:
        entry = json.loads(line)
        pass_in_1 = entry.get("pass_in_1")
        pass_in_5 = entry.get("pass_in_5")
        score1 += 1 if pass_in_1 else 0
        score5 += 1 if pass_in_5 else 0
        token_counts = entry.get("output_token_counts", [])
        token1 += token_counts[0]
        token5 += sum(token_counts)
        cost1 += token_counts[0] * 1.5 / 1000000
        cost5 += sum(token_counts) * 1.5 / 1000000
        cnt += 1
        
    for line in f3:
        entry = json.loads(line)
        pass_in_1 = entry.get("pass_in_1")
        pass_in_5 = entry.get("pass_in_5")
        score1 += 1 if pass_in_1 else 0
        score5 += 1 if pass_in_5 else 0
        token_counts = entry.get("output_token_counts", [])
        token1 += token_counts[0]
        token5 += sum(token_counts)
        cost1 += token_counts[0] * 10 / 1000000
        cost5 += sum(token_counts) * 10 / 1000000
        cnt += 1

    avg_score1 = score1 / cnt
    avg_score5 = score5 / cnt
    avg_cost1 = cost1 / cnt
    avg_cost5 = cost5 / cnt
    avg_token1 = token1 / cnt
    avg_token5 = token5 / cnt
    
    print(f"Average score@1: {avg_score1}")
    print(f"Average score@5: {avg_score5}")
    print(f"Average cost@1: {avg_cost1}")
    print(f"Average cost@5: {avg_cost5}")
    print(f"Average token@1: {avg_token1}")
    print(f"Average token@5: {avg_token5}")

    print("----------------------------------------")
    print("GPT3.5")


data = {}
with open("GPT3.5-output-code_contests.jsonl", "r", encoding='utf-8') as f1, \
     open("GPT3.5-output-HumanEval.jsonl", "r", encoding='utf-8') as f2, \
     open("GPT3.5-output-LeetCode.jsonl", "r", encoding='utf-8') as f3:
    data = {
        json.loads(line)["id"]: json.loads(line) 
        for line in f1
    }
    data.update({
        json.loads(line)["id"]: json.loads(line)
        for line in f2
    })
    data.update({
        json.loads(line)["id"]: json.loads(line)
        for line in f3
    })

with open("baseline_test_predictions.jsonl", "r", encoding='utf-8') as f:
    
    score1 = 0
    score5 = 0
    cost1 = 0
    cost5 = 0
    token1 = 0
    token5 = 0
    cnt = 0

    for line in f:
        entry = json.loads(line)
        if entry["id"] in data:
            data_entry = data[entry["id"]]
            result  = data_entry.get("results", [False]*5)
            score1 += 1 if result[0] else 0
            score5 += 1 if any(result) else 0
            token_counts = data_entry.get("response_tokens", [])
            token1 += token_counts[0]
            token5 += sum(token_counts)
            cost1 += token_counts[0] * 1.5 / 1000000
            cost5 += sum(token_counts) * 1.5 / 1000000
            cnt += 1

    avg_score1 = score1 / cnt
    avg_score5 = score5 / cnt
    avg_cost1 = cost1 / cnt
    avg_cost5 = cost5 / cnt
    avg_token1 = token1 / cnt
    avg_token5 = token5 / cnt
    
    print(f"Average score@1: {avg_score1}")
    print(f"Average score@5: {avg_score5}")
    print(f"Average cost@1: {avg_cost1}")
    print(f"Average cost@5: {avg_cost5}")
    print(f"Average token@1: {avg_token1}")
    print(f"Average token@5: {avg_token5}")

    print("----------------------------------------")
    print("GPT4o")


data = {}
with open("GPT4o-output-code_contests.jsonl", "r", encoding='utf-8') as f1, \
     open("GPT4o-output-HumanEval.jsonl", "r", encoding='utf-8') as f2, \
     open("GPT4o-output-LeetCode.jsonl", "r", encoding='utf-8') as f3:
    data = {
        json.loads(line)["id"]: json.loads(line) 
        for line in f1
    }
    data.update({
        json.loads(line)["id"]: json.loads(line)
        for line in f2
    })
    data.update({
        json.loads(line)["id"]: json.loads(line)
        for line in f3
    })

with open("baseline_test_predictions.jsonl", "r", encoding='utf-8') as f:
    
    score1 = 0
    score5 = 0
    cost1 = 0
    cost5 = 0
    token1 = 0
    token5 = 0
    cnt = 0

    for line in f:
        entry = json.loads(line)
        if entry["id"] in data:
            data_entry = data[entry["id"]]
            result  = data_entry.get("results", [False]*5)
            score1 += 1 if result[0] else 0
            score5 += 1 if any(result) else 0
            token_counts = data_entry.get("response_tokens", [])
            token1 += token_counts[0]
            token5 += sum(token_counts)
            cost1 += token_counts[0] * 10 / 1000000
            cost5 += sum(token_counts) * 10 / 1000000
            cnt += 1

    avg_score1 = score1 / cnt
    avg_score5 = score5 / cnt
    avg_cost1 = cost1 / cnt
    avg_cost5 = cost5 / cnt
    avg_token1 = token1 / cnt
    avg_token5 = token5 / cnt
    
    print(f"Average score@1: {avg_score1}")
    print(f"Average score@5: {avg_score5}")
    print(f"Average cost@1: {avg_cost1}")
    print(f"Average cost@5: {avg_cost5}")
    print(f"Average token@1: {avg_token1}")
    print(f"Average token@5: {avg_token5}")