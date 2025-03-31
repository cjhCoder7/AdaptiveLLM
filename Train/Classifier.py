import json
import joblib
import numpy as np
from transformers import AutoTokenizer, AutoModel
from xgboost import XGBClassifier
import torch
from sklearn.metrics import accuracy_score


class CodeBERTEncoder:
    def __init__(self, model_path="./train"):
        self.device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def encode_batch(self, prompts, batch_size=8):
        all_embeddings = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=128,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

            all_embeddings.append(embeddings)
        return np.concatenate(all_embeddings, axis=0)


def load_dataset(file_path):
    ids = []
    prompts, labels = [], []
    with open(file_path, "r") as f:
        for line in f:
            item = json.loads(line)
            ids.append(item["id"])
            prompts.append(item["prompt"])
            labels.append(item["rank"][0])
    return ids, prompts, np.array(labels)


def train_xgboost_classifier(X, y):
    return XGBClassifier(
        objective="multi:softmax",
        num_class=len(np.unique(y)),
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        random_state=42,
    ).fit(X, y)


def save_predictions(predictions, file_path):
    with open(file_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")


def main():
    encoder = CodeBERTEncoder()

    train_ids, train_prompts, train_labels = load_dataset("train_data.jsonl")
    test_ids, test_prompts, test_labels = load_dataset("test_data.jsonl")

    X_train = encoder.encode_batch(train_prompts)
    X_test = encoder.encode_batch(test_prompts)

    print("Train labels unique:", np.unique(train_labels))
    print("Test labels unique:", np.unique(test_labels))

    classifier = train_xgboost_classifier(X_train, train_labels)
    joblib.dump(classifier, "xgboost_model_1.pkl")

    test_preds = classifier.predict(X_test)
    test_probs = classifier.predict_proba(X_test)

    accuracy = accuracy_score(test_labels, test_preds)
    print(f"Test set accuracy: {accuracy * 100:.2f}%")

    predictions = [
        {
            "id": id,
            "prompt": prompt,
            "predicted_rank": int(pred),
            "probability": prob.tolist(),
            "true_rank": int(true),
        }
        for id, prompt, pred, prob, true in zip(
            test_ids, test_prompts, test_preds, test_probs, test_labels
        )
    ]

    save_predictions(predictions, "predictions_1.jsonl")


if __name__ == "__main__":
    main()
