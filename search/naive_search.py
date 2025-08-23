import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple
from collections import Counter


"""
Algorithms:
- Brute-Force Vector Search
- KNN Search
- KMeans Search

TODO:
- Implement all algorithms
- Write sample to confirm they work
- Create classes for other implementations by professionals
"""

########################################################################################
# Brute-Force Vector Search
########################################################################################


class BruteForceSearch:
    def __init__(self):
        self.dataset = None

    def train(self, dataset: np.ndarray):
        self.dataset = dataset

    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[float, np.ndarray]]:
        distances = np.linalg.norm(self.dataset - query_vector, axis=1)
        indices = np.argsort(distances)[:k]
        return [(distances[i], self.dataset[i]) for i in indices]


########################################################################################
# KNN Search
########################################################################################

class KNNSearch:
    def __init__(self):
        self.dataset = None
        self.labels = None

    def train(self, dataset: List[np.ndarray], labels: List[str] = None):
        self.dataset = dataset
        self.labels = labels

    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[float, np.ndarray]]:
        distances = []
        for vector in self.dataset:
            distance = np.linalg.norm(query_vector - vector)
            distances.append((distance, vector))
        return sorted(distances, key=lambda x: x[0])[:k]

    def classify(self, query_vector: np.ndarray, k: int) -> str:
        if self.labels is None:
            raise ValueError("Labels not provided during training. Cannot perform classification.")
        neighbors = self.search(query_vector, k)
        neighbor_labels = [self.labels[self.dataset.index(vector)] for _, vector in neighbors]
        return Counter(neighbor_labels).most_common(1)[0][0]


########################################################################################
# KMeans Search
########################################################################################

class KMeansSearch:
    def __init__(self, k_clusters: int, max_iterations: int = 100):
        self.k_clusters = k_clusters
        self.max_iterations = max_iterations
        self.dataset = None
        self.centroids = None
        self.cluster_assignments = None

    def train(self, dataset: np.ndarray):
        self.dataset = dataset
        self.centroids, self.cluster_assignments = self._kmeans_clustering()

    def _kmeans_clustering(self) -> Tuple[np.ndarray, np.ndarray]:
        centroids = self.dataset[np.random.choice(self.dataset.shape[0], self.k_clusters, replace=False)]

        for _ in range(self.max_iterations):
            distances = np.array([np.linalg.norm(self.dataset - c, axis=1) for c in centroids])
            cluster_assignments = np.argmin(distances, axis=0)

            new_centroids = np.array([self.dataset[cluster_assignments == i].mean(axis=0) for i in range(self.k_clusters)])

            if np.all(centroids == new_centroids):
                break

            centroids = new_centroids

        return centroids, cluster_assignments

    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[float, np.ndarray]]:
        centroid_distances = [np.linalg.norm(query_vector - centroid) for centroid in self.centroids]
        nearest_centroid = np.argmin(centroid_distances)

        cluster_points = self.dataset[self.cluster_assignments == nearest_centroid]

        distances = [np.linalg.norm(query_vector - point) for point in cluster_points]
        sorted_indices = np.argsort(distances)[:k]

        return [(distances[i], cluster_points[i]) for i in sorted_indices]

def test_algorithms():
    # Generate sample data
    np.random.seed(42)
    dataset = np.random.rand(1000, 128)
    query = np.random.rand(128)

    print("Testing Brute-Force Vector Search:")
    brute_force = BruteForceSearch()
    brute_force.train(dataset)
    results = brute_force.search(query, k=5)
    print("Top 5 results (distance, vector):")
    for distance, vector in results:
        print(f"Distance: {distance:.4f}, Vector: {vector[:5]}...")

    print("\nTesting KNN Search:")
    knn = KNNSearch()
    knn.train(dataset.tolist())
    results = knn.search(query, k=5)
    print("Top 5 results (distance, vector):")
    for distance, vector in results:
        print(f"Distance: {distance:.4f}, Vector: {vector[:5]}...")

    print("\nTesting KMeans Search:")
    kmeans = KMeansSearch(k_clusters=10)
    kmeans.train(dataset)
    results = kmeans.search(query, k=5)
    print("Top 5 results (distance, vector):")
    for distance, vector in results:
        print(f"Distance: {distance:.4f}, Vector: {vector[:5]}...")

if __name__ == "__main__":
    test_algorithms()