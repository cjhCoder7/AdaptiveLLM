import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator


def load_data(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            item = json.loads(line)
            data.append(float(item["median"]))
    return np.array(data).reshape(-1, 1)


X = load_data("DeepSeek-R1-Distill-Qwen-32B-output-total-median.jsonl")

k_range = range(1, 11)
inertias = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

kneedle = KneeLocator(
    x=k_range,
    y=inertias,
    curve="convex", 
    direction="decreasing", 
    interp_method="interp1d",  
)
optimal_k = kneedle.elbow

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(k_range, inertias, "bo-", markersize=8)
plt.xlabel("Number of clusters (K)", fontsize=12)
plt.ylabel("Inertia", fontsize=12)
plt.title("Elbow Method Analysis", fontsize=14)

if optimal_k is not None:
    plt.scatter(
        optimal_k,
        inertias[optimal_k - 1],
        s=200,
        facecolors="none",
        edgecolors="r",
        linewidths=2,
    )
    plt.text(
        optimal_k + 0.2,
        inertias[optimal_k - 1] * 0.95,
        f"Optimal K = {optimal_k}",
        fontsize=12,
        color="red",
    )

if optimal_k is not None:
    final_kmeans = KMeans(
        n_clusters=optimal_k, init="k-means++", random_state=42, n_init=10
    )
    cluster_labels = final_kmeans.fit_predict(X)

    sorted_centers_indices = np.argsort(final_kmeans.cluster_centers_.flatten())
    center_mapping = {
        old_label: new_label
        for new_label, old_label in enumerate(sorted_centers_indices)
    }
    new_cluster_labels = np.array([center_mapping[label] for label in cluster_labels])

    sorted_centers = final_kmeans.cluster_centers_[sorted_centers_indices]

    plt.subplot(1, 2, 2)
    for i in range(optimal_k):
        cluster_data = X[new_cluster_labels == i]
        plt.scatter(
            cluster_data, np.zeros_like(cluster_data), label=f"Cluster {i}", alpha=0.6
        )
    plt.xlabel("Value", fontsize=12)
    plt.yticks([])
    plt.title(f"K={optimal_k} Clustering Result", fontsize=14)
    plt.legend()

plt.tight_layout()
plt.show()
plt.savefig("K_means.png")

if optimal_k is not None:
    print(f"\nAutomatically detected optimal K value: {optimal_k}")
    print("Cluster centers:", sorted_centers.flatten())

    for i in range(optimal_k):
        cluster_data = X[new_cluster_labels == i]
        print(f"\nCluster {i}:")
        print(f" - Sample count: {len(cluster_data)}")
        print(f" - Value range: {cluster_data.min():.1f} ~ {cluster_data.max():.1f}")
        print(f" - Mean: {cluster_data.mean():.1f}")
        print(f" - Standard deviation: {cluster_data.std():.1f}")

    with open("DeepSeek-R1-Distill-Qwen-32B-output-total-median.jsonl", "r") as f, open(
        "DeepSeek-R1-Distill-Qwen-32B-output-total-median_with_clusters.jsonl",
        "w",
    ) as out_f:
        for idx, line in enumerate(f):
            item = json.loads(line)
            item["cluster"] = int(new_cluster_labels[idx])
            out_f.write(json.dumps(item) + "\n")

    print("\nClassification results have been added to the file.")
else:
    print("No obvious elbow point detected, please check data distribution")