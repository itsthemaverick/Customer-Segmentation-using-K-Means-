import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

def plot_elbow(k_values, inertias, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    k_values = np.array(k_values).flatten()
    inertias = np.array(inertias).flatten()
    if len(k_values) != len(inertias):
        raise ValueError(f"x and y must have same length. Got {len(k_values)} and {len(inertias)}")
    plt.figure()
    plt.plot(k_values, inertias, marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal k")
    plt.savefig(save_path)
    plt.close()

def plot_clusters_2d(features, labels, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    features = np.array(features)
    labels = np.array(labels).flatten()
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)
    plt.figure()
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Customer Clusters (PCA Projection)")
    plt.savefig(save_path)
    plt.close()
