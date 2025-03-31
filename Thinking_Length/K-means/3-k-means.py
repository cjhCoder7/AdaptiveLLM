import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def load_data(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            item = json.loads(line)
            data.append(float(item["median"]))
    return np.array(data).reshape(-1, 1)


file_path = "DeepSeek-R1-Distill-Qwen-32B-output-total-median.jsonl"
X = load_data(file_path)

final_kmeans = KMeans(n_clusters=3, init="k-means++", random_state=42, n_init=10)
cluster_labels = final_kmeans.fit_predict(X)

centers = final_kmeans.cluster_centers_.flatten()
sorted_indices = np.argsort(centers)
label_mapping = {
    old_label: new_label for new_label, old_label in enumerate(sorted_indices)
}

new_cluster_labels = np.array([label_mapping[label] for label in cluster_labels])

plt.figure(figsize=(12, 6))

for i in range(3):
    cluster_data = X[new_cluster_labels == i]
    plt.scatter(
        cluster_data, np.zeros_like(cluster_data), label=f"Cluster {i}", alpha=0.6
    )
plt.xlabel("Value", fontsize=12)
plt.yticks([])
plt.title(f"K=3 Clustering Result", fontsize=14)
plt.legend()

plt.tight_layout()
plt.show()
plt.savefig(
    "DeepSeek-R1-Distill-Qwen-32B-output-total-median_kmeans_clustering_result.png"
)
plt.close()

print(f"\nNum: 3")
print("The center of clustering:", final_kmeans.cluster_centers_.flatten())

output_file_path = file_path.replace(".jsonl", "_with_clusters.jsonl")
with open(file_path, "r") as f, open(output_file_path, "w") as out_f:
    for idx, line in enumerate(f):
        item = json.loads(line)
        item["cluster"] = int(new_cluster_labels[idx]) 
        out_f.write(json.dumps(item) + "\n")
