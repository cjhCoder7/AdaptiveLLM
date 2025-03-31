import json
from sklearn.metrics import (
    adjusted_rand_score,
    fowlkes_mallows_score,
    mutual_info_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def load_clusters(file_path):
    clusters = []
    with open(file_path, "r") as f:
        for line in f:
            item = json.loads(line)
            clusters.append(item["cluster"])
    return clusters


file_path_1 = "DeepSeek-R1-Distill-Qwen-1.5B-output-total-median_with_clusters.jsonl"
file_path_2 = "DeepSeek-R1-Distill-Qwen-7B-output-total-median_with_clusters.jsonl"

clusters_1 = load_clusters(file_path_1)
clusters_2 = load_clusters(file_path_2)

ari = adjusted_rand_score(clusters_1, clusters_2)
fmi = fowlkes_mallows_score(clusters_1, clusters_2)
mi = mutual_info_score(clusters_1, clusters_2)
nmi = normalized_mutual_info_score(clusters_1, clusters_2)
homogeneity = homogeneity_score(clusters_1, clusters_2)
completeness = completeness_score(clusters_1, clusters_2)

print(f"调整兰德指数 (ARI): {ari:.4f}")
print(f"Fowlkes - Mallows 指数 (FMI): {fmi:.4f}")
print(f"互信息 (MI): {mi:.4f}")
print(f"归一化互信息 (NMI): {nmi:.4f}")
print(f"同质性: {homogeneity:.4f}")
print(f"完整性: {completeness:.4f}")

cm = confusion_matrix(clusters_1, clusters_2)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("DeepSeek-R1-Distill-Qwen-7B_clusters")
plt.ylabel("DeepSeek-R1-Distill-Qwen-1.5B_clusters")
plt.title("Confusion Matrix")
plt.show()
plt.savefig("1.5B-7B-confusion_matrix.png")
