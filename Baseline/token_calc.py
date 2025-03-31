import json
import tiktoken

def process_file(input_path):  
    encoder = tiktoken.get_encoding("cl100k_base")
    
    with open(input_path, 'r+', encoding='utf-8') as file:  
        lines = file.readlines()
        file.seek(0)  
        file.truncate()  
        
        for line in lines:
            entry = json.loads(line)
            if 'response' in entry:
                token_counts = [len(encoder.encode(text)) for text in entry['response']]
                entry['response_tokens'] = token_counts
            file.write(json.dumps(entry) + '\n')

process_file(
    "GPT4o-output-LeetCode.jsonl"
)