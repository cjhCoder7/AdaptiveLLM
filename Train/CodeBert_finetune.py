import json
import torch
import os
import numpy as np
from collections import defaultdict, Counter
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
import torch.nn as nn
import torch.optim as optim

# from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "6, 7"
MODEL_NAME = "./codebert-base"
BATCH_SIZE = 32
EPOCHS = 1
MARGIN = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 128

# print(f"Using device: {DEVICE}")
# if torch.cuda.is_available():
#     print(f"Current device: {torch.cuda.current_device()}")


class TripletGenerator:

    def __init__(self, data_path):
        with open(data_path) as f:
            self.data = [json.loads(line) for line in f]

        self.prompt_to_cluster = {item["prompt"]: item["cluster"] for item in self.data}

        self.cluster_dict = defaultdict(list)
        for idx, item in enumerate(self.data):
            self.cluster_dict[item["cluster"]].append(idx)

    def generate_triplets(self):
        triplets = []
        for anchor_idx, anchor in enumerate(self.data):
            cluster = anchor["cluster"]

            positives = self.cluster_dict[cluster].copy()
            positives.remove(anchor_idx)
            if not positives:
                continue
            positive_idx = np.random.choice(positives)

            negative_clusters = [c for c in self.cluster_dict.keys() if c != cluster]
            if not negative_clusters:
                continue
            neg_cluster = np.random.choice(negative_clusters)
            negative_idx = np.random.choice(self.cluster_dict[neg_cluster])

            triplets.append(
                {
                    "anchor": anchor["prompt"],
                    "positive": self.data[positive_idx]["prompt"],
                    "negative": self.data[negative_idx]["prompt"],
                    "cluster": cluster,
                }
            )
        return triplets


def print_distribution(data, prompt_to_cluster, name):
    clusters = [prompt_to_cluster[item["anchor"]] for item in data]
    counter = Counter(clusters)
    total = len(data)

    print(f"{name} Distribution (Total{total}):")
    for cluster in sorted(counter):
        count = counter[cluster]
        print(f"Cluster {cluster}: {count} ({count/total:.1%})")


class TripletDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor, positive, negative = self.data[idx].texts
        anchor_encoding = self.tokenizer(
            anchor,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        positive_encoding = self.tokenizer(
            positive,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        negative_encoding = self.tokenizer(
            negative,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        anchor_input_ids = anchor_encoding["input_ids"].squeeze(0)
        anchor_attention_mask = anchor_encoding["attention_mask"].squeeze(0)

        positive_input_ids = positive_encoding["input_ids"].squeeze(0)
        positive_attention_mask = positive_encoding["attention_mask"].squeeze(0)

        negative_input_ids = negative_encoding["input_ids"].squeeze(0)
        negative_attention_mask = negative_encoding["attention_mask"].squeeze(0)

        return {
            "anchor_input_ids": anchor_input_ids,
            "anchor_attention_mask": anchor_attention_mask,
            "positive_input_ids": positive_input_ids,
            "positive_attention_mask": positive_attention_mask,
            "negative_input_ids": negative_input_ids,
            "negative_attention_mask": negative_attention_mask,
        }


class CodeBERTTripletModel(nn.Module):
    def __init__(self, model_name):
        super(CodeBERTTripletModel, self).__init__()
        self.codebert = RobertaModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings


generator = TripletGenerator(
    "DeepSeek-R1-Distill-Qwen-32B-output-total-median_with_clusters.jsonl"
)
triplet_data = generator.generate_triplets()
prompt_to_cluster = generator.prompt_to_cluster

from sentence_transformers import InputExample

triplet_examples = [
    InputExample(
        texts=[item["anchor"], item["positive"], item["negative"]],
        label=item["cluster"],
    )
    for item in triplet_data
]

print("=" * 50)
print_distribution(triplet_data, prompt_to_cluster, "Original Dataset") 
print("=" * 50)

tokenizer = RobertaTokenizer.from_pretrained(
    MODEL_NAME, clean_up_tokenization_spaces=True
)
model = CodeBERTTripletModel(MODEL_NAME)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs.")
    model = nn.DataParallel(model)
model.to(DEVICE)

train_dataset = TripletDataset(triplet_examples, tokenizer, MAX_LENGTH)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)


criterion = nn.TripletMarginWithDistanceLoss(
    distance_function=nn.CosineSimilarity(dim=1), margin=MARGIN
)
optimizer = optim.Adam(model.parameters(), lr=2e-5)


for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        anchor_input_ids = batch["anchor_input_ids"].to(DEVICE)
        anchor_attention_mask = batch["anchor_attention_mask"].to(DEVICE)
        positive_input_ids = batch["positive_input_ids"].to(DEVICE)
        positive_attention_mask = batch["positive_attention_mask"].to(DEVICE)
        negative_input_ids = batch["negative_input_ids"].to(DEVICE)
        negative_attention_mask = batch["negative_attention_mask"].to(DEVICE)

        optimizer.zero_grad()

        anchor_output = model(anchor_input_ids, anchor_attention_mask)
        positive_output = model(positive_input_ids, positive_attention_mask)
        negative_output = model(negative_input_ids, negative_attention_mask)

        loss = criterion(anchor_output, positive_output, negative_output)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    # # 
    # model.eval()
    # total_val_loss = 0
    # with torch.no_grad():
    #     for batch in val_loader:
    #         anchor_input_ids = batch["anchor_input_ids"].to(DEVICE)
    #         anchor_attention_mask = batch["anchor_attention_mask"].to(DEVICE)
    #         positive_input_ids = batch["positive_input_ids"].to(DEVICE)
    #         positive_attention_mask = batch["positive_attention_mask"].to(DEVICE)
    #         negative_input_ids = batch["negative_input_ids"].to(DEVICE)
    #         negative_attention_mask = batch["negative_attention_mask"].to(DEVICE)

    #         anchor_output = model(anchor_input_ids, anchor_attention_mask)
    #         positive_output = model(positive_input_ids, positive_attention_mask)
    #         negative_output = model(negative_input_ids, negative_attention_mask)

    #         loss = criterion(anchor_output, positive_output, negative_output)
    #         total_val_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {total_train_loss / len(train_loader)}"
    )
    print(f"Current LR: {optimizer.param_groups[0]['lr']}")

if isinstance(model, nn.DataParallel):
    model.module.codebert.save_pretrained("./train")
    tokenizer.save_pretrained("./train")
else:
    model.codebert.save_pretrained("./train")
    tokenizer.save_pretrained("./train")
print("Finished training.")
