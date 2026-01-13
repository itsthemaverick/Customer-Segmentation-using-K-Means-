from sklearn.cluster import KMeans

def run_kmeans(features, number_of_clusters):
    model = KMeans(n_clusters=number_of_clusters, random_state=42)
    model.fit(features)
    return model.labels_, model.cluster_centers_
