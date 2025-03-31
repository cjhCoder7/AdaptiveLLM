import json  
from openai import OpenAI

client = OpenAI(
    api_key="API_KEY",
    base_url="https://api.siliconflow.cn/v1"
)

with open('baseline_test.jsonl', 'r', encoding='utf-8') as infile, \
     open('baseline_test_predictions.jsonl', 'w', encoding='utf-8') as outfile:

    for line in infile:
        data = json.loads(line)
        
        messages = [
            {"role": "system", "content": "Classify the problem to 5 degrees according to its difficulty"},
            {"role": "user", "content": data['prompt']},
        ]

        response = client.chat.completions.create(
            model="ft:LoRA/Qwen/Qwen2.5-7B-Instruct:qanvrd13kq:llm:kirxgblyynxmfbypyele",
            messages=messages,
            stream=True,
            max_tokens=4096
        )

        current_response = ""
        for chunk in response:
            content = chunk.choices[0].delta.content or ""
            print(content, end='')
            current_response += content
        
        data['prediction'] = current_response
        outfile.write(json.dumps(data, ensure_ascii=False) + '\n')