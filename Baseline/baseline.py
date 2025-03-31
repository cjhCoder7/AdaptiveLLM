import json

def process_jsonl_files(file1_path, file2_path, file3_path, output_path):
    with open(file1_path, 'r') as f1, \
         open(file2_path, 'r') as f2, \
         open(file3_path, 'r') as f3, \
         open(output_path, 'a') as out_file:

        for line1, line2, line3 in zip(f1, f2, f3):
            data1 = json.loads(line1.strip())
            data2 = json.loads(line2.strip())
            data3 = json.loads(line3.strip())

            pass_rate1 = data1['pass_rate']
            pass_rate2 = data2['pass_rate']
            pass_rate3 = data3['pass_rate']

            if pass_rate1 == 1 or pass_rate1 + pass_rate2 >= 1.4:
                new_entry = {
                    "id": data1['id'],
                    "prompt": data1['prompt'],
                    "cluster": 1
                }
                out_file.write(json.dumps(new_entry) + '\n')
            elif pass_rate2 == 1:
                new_entry = {
                    "id": data1['id'],
                    "prompt": data1['prompt'],
                    "cluster": 2
                }
                out_file.write(json.dumps(new_entry) + '\n')
            elif pass_rate3 == 1:
                new_entry = {
                    "id": data1['id'],
                    "prompt": data1['prompt'],
                    "cluster": 3
                }
                out_file.write(json.dumps(new_entry) + '\n')
            elif pass_rate2 > 0.4 or pass_rate3 > 0.4:
                new_entry = {
                    "id": data1['id'],
                    "prompt": data1['prompt'],
                    "cluster": 4
                }
                out_file.write(json.dumps(new_entry) + '\n')            
            else:
                new_entry = {
                    "id": data1['id'],
                    "prompt": data1['prompt'],
                    "cluster": 5
                }
                out_file.write(json.dumps(new_entry) + '\n')


if __name__ == "__main__":
    process_jsonl_files(
        '../Result/CodeLlama-7b-Instruct-hf-output-HumanEval.jsonl',
        'GPT3.5-output-HumanEval.jsonl',
        'GPT4o-output-HumanEval.jsonl',
        'baseline.jsonl'
    )

    process_jsonl_files(
        '../Result/CodeLlama-7b-Instruct-hf-output-LeetCode.jsonl',
        'GPT3.5-output-LeetCode.jsonl',
        'GPT4o-output-LeetCode.jsonl',
        'baseline.jsonl'
    )

    process_jsonl_files(
        '../Result/CodeLlama-7b-Instruct-hf-output-CodeContests.jsonl',
        'GPT3.5-output-CodeContests.jsonl',
        'GPT4o-output-CodeContests.jsonl',
        'baseline.jsonl'
    )