import json
from pathlib import Path

MODEL_MAP = {
    (1, 2): "CodeLlama",
    (3, 4): "GPT3.5"
}

DATASET_MAP = {
    "HumanEval": "HumanEval.jsonl",
    "LeetCode": "LeetCode.jsonl",
    "code_contests": "code_contests.jsonl"
}

model_data = {}
for model in ["CodeLlama", "GPT3.5", "GPT4o"]:
    for dataset in DATASET_MAP.values():
        path = Path(f"{model}-output-{dataset}")
        if path.exists():
            with open(path, encoding='utf-8') as f: 
                model_data[(model, dataset)] = {
                    json.loads(line)["id"]: json.loads(line) 
                    for line in f
                }

with open("baseline_test_predictions.jsonl", "r", encoding='utf-8') as infile, \
     open("CodeLlama_processed_results.jsonl", "w", encoding='utf-8') as cl_outfile, \
     open("GPT3.5_processed_results.jsonl", "w", encoding='utf-8') as gpt3_outfile, \
     open("GPT4o_processed_results.jsonl", "w", encoding='utf-8') as gpt4_outfile:

    model_files = {
        "CodeLlama": cl_outfile,
        "GPT3.5": gpt3_outfile,
        "GPT4o": gpt4_outfile
    }

    for line in infile:
        data = json.loads(line)
        prediction = data["prediction"]
        
        model = "GPT4o"
        try:
            pred_num = int(prediction)
            for k, v in MODEL_MAP.items():
                if pred_num in k:
                    model = v
                    break
        except ValueError:
            pass

        dataset = data["id"].split("/")[0]
        file_suffix = DATASET_MAP.get(dataset, "")
        
        key = (model, file_suffix)
        if key not in model_data:
            continue
            
        result_entry = model_data[key].get(data["id"], {})
        results = result_entry.get("results", [False]*5)
        
        data["pass_in_1"] = bool(results[0]) if results else False
        data["pass_in_5"] = any(results) if results else False
        
        if model == "CodeLlama":
            token_counts = result_entry.get("output_token_counts", [])
            data["output_token_counts"] = token_counts
        else :
            token_counts = result_entry.get("response_tokens", [])
            data["output_token_counts"] = token_counts
        
        model_files[model].write(json.dumps(data) + "\n")