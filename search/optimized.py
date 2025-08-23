"""
This file will take all of the naive and high performance search algorithms and create optimized versions of them.
    I am not writing these optimized versions, I am going to use the best libraries I can find for each algorithm.
    I will put them into the same format as the naive versions so that the benchmarking script can run them.

"""
import hnswlib
import faiss
import numpy as np
from typing import List, Tuple

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

class HNSW:
    def __init__(self, dim: int, space: str = 'l2'):
        self.dim = dim
        self.space = space
        self.index = None

    def build(self, dataset: np.ndarray, ef_construction: int = 200, M: int = 16):
        num_elements = dataset.shape[0]
        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M)
        self.index.add_items(dataset)

    def search(self, query: np.ndarray, k: int) -> List[Tuple[float, np.ndarray]]:
        self.index.set_ef(max(k, 50))  # ef should always be > k
        labels, distances = self.index.knn_query(query, k=k)
        return labels.flatten(), distances.flatten()

class FAISS:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = None

    def build(self, dataset: np.ndarray):
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(dataset.astype('float32'))

    def search(self, query: np.ndarray, k: int) -> List[Tuple[float, np.ndarray]]:
        distances, labels = self.index.search(query.astype('float32'), k)
        return labels.flatten(), distances.flatten()

class FaissPQSearch:
    def __init__(self, dim: int, m: int = 8, nbits: int = 8):
        self.dim = dim
        self.m = m
        self.nbits = nbits
        self.index = None

    def build(self, dataset: np.ndarray):
        self.index = faiss.IndexPQ(self.dim, self.m, self.nbits)
        self.index.train(dataset)
        self.index.add(dataset)

    def search(self, query: np.ndarray, k: int) -> List[Tuple[float, np.ndarray]]:
        if len(query.shape) == 1:
            query = query.reshape(1, -1)
        
        distances, indices = self.index.search(query, k)
        return indices.flatten(), distances.flatten()
    
class SklearnKNN:
    def __init__(self, n_neighbors: int = 5, algorithm: str = 'auto'):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.index = None

    def build(self, dataset: np.ndarray):
        self.index = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm=self.algorithm)
        self.index.fit(dataset)

    def search(self, query: np.ndarray, k: int) -> List[Tuple[float, np.ndarray]]:
        if len(query.shape) == 1:
            query = query.reshape(1, -1)
        distances, indices = self.index.kneighbors(query, n_neighbors=k)
        return indices.flatten(), distances.flatten()

class SklearnKMeans:
    def __init__(self, n_clusters: int = 10):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.dataset = None

    def build(self, dataset: np.ndarray):
        self.kmeans = KMeans(n_clusters=self.n_clusters)
        self.kmeans.fit(dataset)
        self.dataset = dataset

    def search(self, query: np.ndarray, k: int) -> List[Tuple[float, np.ndarray]]:
        if len(query.shape) == 1:
            query = query.reshape(1, -1)
        
        # Find the nearest cluster center
        cluster_label = self.kmeans.predict(query)
        
        # Get all points in the same cluster
        cluster_points = self.dataset[self.kmeans.labels_ == cluster_label[0]]
        
        # Compute distances to all points in the cluster
        distances = np.linalg.norm(cluster_points - query, axis=1)
        
        # Sort and return top k results
        k = min(k, len(cluster_points))
        indices = np.argsort(distances)[:k]
        return indices, distances[indices]

class BruteForceSearch:
    def __init__(self):
        self.dataset = None

    def build(self, dataset: np.ndarray):
        self.dataset = dataset

    def search(self, query: np.ndarray, k: int) -> List[Tuple[float, np.ndarray]]:
        if len(query.shape) == 1:
            query = query.reshape(1, -1)
        
        distances = np.linalg.norm(self.dataset - query, axis=1)
        indices = np.argsort(distances)[:k]
        return indices, distances[indices]

if __name__ == "__main__":
    # Test HNSW
    print("Testing HNSW:")
    data = np.float32(np.random.random((10000, 128)))
    query = np.float32(np.random.random((1, 128)))
    hnsw = HNSW(dim=128, space='l2')
    hnsw.build(data)
    print(hnsw.search(query, k=10))

    # Test FAISS
    print("\nTesting FAISS:")
    faiss_index = FAISS(dim=128)
    faiss_index.build(data)
    print(faiss_index.search(query, k=10))

    # Test FaissPQSearch
    print("\nTesting FaissPQSearch:")
    pq_search = FaissPQSearch(dim=128)
    pq_search.build(data)
    print(pq_search.search(query, k=10))

    # Test SklearnKNN
    print("Testing SklearnKNN:")
    knn = SklearnKNN(n_neighbors=10)
    knn.build(data)
    indices, distances = knn.search(query, k=10)
    print(f"Indices: {indices}")
    print(f"Distances: {distances}")

    # Test SklearnKMeans
    print("\nTesting SklearnKMeans:")
    kmeans = SklearnKMeans(n_clusters=100)
    kmeans.build(data)
    indices, distances = kmeans.search(query, k=10)
    print(f"Indices: {indices}")
    print(f"Distances: {distances}")

    # Test BruteForceSearch
    print("\nTesting BruteForceSearch:")
    brute_force = BruteForceSearch()
    brute_force.build(data)
    indices, distances = brute_force.search(query, k=10)
    print(f"Indices: {indices}")
    print(f"Distances: {distances}")