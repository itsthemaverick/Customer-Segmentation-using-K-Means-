from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

def compute_elbow(features, k_values):
    inertias = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(features)
        inertias.append(float(model.inertia_))
    return np.array(inertias).flatten()

def compute_silhouette(features, labels):
    return float(silhouette_score(features, labels))
