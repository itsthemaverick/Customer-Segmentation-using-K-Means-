from src.load_data import load_data
from src.preprocess import preprocess_data
from src.clustering import run_kmeans
from src.metrics import compute_elbow, compute_silhouette
from src.visualize import plot_elbow, plot_clusters_2d

def main():
    data_path = "data/Mall_Customers.csv"
    elbow_plot_path = "visualizations/elbow.png"
    cluster_plot_path = "visualizations/clusters.png"

    data = load_data(data_path)
    features = preprocess_data(data)

    k_values = range(2, 11)
    inertias = compute_elbow(features, k_values)
    plot_elbow(k_values, inertias, elbow_plot_path)

    chosen_k = 4
    labels, centers = run_kmeans(features, chosen_k)
    silhouette = compute_silhouette(features, labels)
    plot_clusters_2d(features, labels, cluster_plot_path)

    print(f"Chosen number of clusters (k): {chosen_k}")
    print(f"Silhouette Score: {silhouette:.3f}")

if __name__ == "__main__":
    main()

