import json

def load_ids(file_path):
    ids = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            ids.add(data['id'])
    return ids

train_ids = load_ids('../Train/train_data.jsonl')

with open('baseline.jsonl', 'r', encoding='utf-8') as baseline, \
     open('baseline_train.jsonl', 'w', encoding='utf-8') as out_train, \
     open('baseline_test.jsonl', 'w', encoding='utf-8') as out_test:

    for line in baseline:
        data = json.loads(line)
        if data['id'] in train_ids:
            out_train.write(json.dumps(data) + '\n')
        else:
            out_test.write(json.dumps(data) + '\n')