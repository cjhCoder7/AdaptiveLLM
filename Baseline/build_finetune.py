import json

with open('baseline_train.jsonl', 'r', encoding='utf-8') as infile, \
     open('baseline_train_finetune.jsonl', 'w', encoding='utf-8') as outfile:

    for line in infile:
        data = json.loads(line)
        filtered_data = {
            "messages": [
                {
                    "role": "system", 
                    "content": "Classify the problem to 5 degrees according to its difficulty"
                },
                {
                    "role": "user",
                    "content": data['prompt']
                },
                {
                    "role": "assistant",
                    "content": str(data['cluster'])
                }
            ]
        }
        outfile.write(json.dumps(filtered_data) + '\n')