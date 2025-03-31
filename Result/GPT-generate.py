import requests
import json
import os

url = ""
headers = {
   'Authorization': 'API-KEY',
   'Content-Type': 'application/json'
}

input_file = os.path.join('Processed Data', 'prompts.jsonl')
output_file = os.path.join('Processed Data', 'GPT4o-output-LeetCode.jsonl')

processed_ids = set()
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as outfile:
        for line in outfile:
            data = json.loads(line)
            processed_ids.add(data['id'])

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'a', encoding='utf-8') as outfile:
    for line in infile:
        data = json.loads(line)
        
        if data['id'] in processed_ids:
            continue
            
        payload = json.dumps({
            "model": "gpt-4o",
            "messages": [
               {
                  "role": "system",
                  "content": "You are a helpful assistant. Complete the code according to the prompt."
               },
               {
                  "role": "user",
                  "content": data['prompt']
               }
            ],
            "max_tokens": 2048,
            "n": 5
         })
        
        response = requests.request("POST", url, headers=headers, data=payload)
        #print(response.text)
        response_data = json.loads(response.text)
        
        responses = [choice['message']['content'].strip() for choice in response_data['choices']]
        data['response'] = responses
        
        outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
        print(data["id"])
        

# import requests
# import json

# url = ""

# payload = json.dumps({
#    "model": "gpt-3.5-turbo",
#    "messages": [
#       {
#          "role": "system",
#          "content": "You are a helpful assistant."
#       },
#       {
#          "role": "user",
#          "content": "Hello!"
#       }
#    ]
# })
# headers = {
#    'Authorization': 'API-KEY',
#    'Content-Type': 'application/json'
# }

# response = requests.request("POST", url, headers=headers, data=payload)

# print(response.text)
