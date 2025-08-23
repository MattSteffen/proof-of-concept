import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
import heapq

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


"""
Algorithms:
- Faiss
- ScaNN
- HNSW
- Product Quantization
- NGT-onng
"""

########################################################################################
# Faiss
########################################################################################

class SimpleFAISS:
    def __init__(self, n_subvectors: int, n_clusters: int):
        self.n_subvectors = n_subvectors
        self.n_clusters = n_clusters
        self.centroids = None
        self.encoded_dataset = None
        self.dataset = None

    def train(self, dataset: np.ndarray):
        self.dataset = dataset
        vector_dim = dataset.shape[1]
        subvector_dim = vector_dim // self.n_subvectors

        self.centroids = []
        for i in range(self.n_subvectors):
            start = i * subvector_dim
            end = (i + 1) * subvector_dim
            subvectors = dataset[:, start:end]

            kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)
            kmeans.fit(subvectors)
            self.centroids.append(kmeans.cluster_centers_)

        self.encoded_dataset = self._encode_vectors(dataset)

    def _encode_vectors(self, vectors: np.ndarray) -> np.ndarray:
        encoded = np.zeros((vectors.shape[0], self.n_subvectors), dtype=int)
        subvector_dim = vectors.shape[1] // self.n_subvectors

        for i in range(self.n_subvectors):
            start = i * subvector_dim
            end = (i + 1) * subvector_dim
            subvectors = vectors[:, start:end]

            distances = np.linalg.norm(subvectors[:, np.newaxis, :] - self.centroids[i], axis=2)
            encoded[:, i] = np.argmin(distances, axis=1)

        return encoded

    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[float, np.ndarray]]:
        encoded_query = self._encode_vectors(query_vector.reshape(1, -1))[0]
        distances = np.zeros(len(self.encoded_dataset))

        for i in range(self.n_subvectors):
            distances += (self.encoded_dataset[:, i] != encoded_query[i]).astype(int)

        top_k_indices = np.argsort(distances)[:k]
        results = [(distances[i], self.dataset[i]) for i in top_k_indices]
        return results



########################################################################################
# ScaNN
########################################################################################
# ScaNN Implementation
class ScaNN:
    def __init__(self, num_leaves, num_clusters_per_block):
        self.num_leaves = num_leaves
        self.num_clusters_per_block = num_clusters_per_block
        self.tree = None
        self.quantizers = {}

    def build_partition_tree(self, dataset):
        kmeans = KMeans(n_clusters=self.num_leaves)
        labels = kmeans.fit_predict(dataset)
        self.tree = defaultdict(list)
        for i, label in enumerate(labels):
            self.tree[label].append(i)
        return kmeans.cluster_centers_

    def train_vector_quantizer(self, leaf_data):
        kmeans = KMeans(n_clusters=self.num_clusters_per_block)
        kmeans.fit(leaf_data)
        return kmeans

    def train(self, dataset):
        self.centroids = self.build_partition_tree(dataset)
        for leaf, indices in self.tree.items():
            leaf_data = dataset[indices]
            self.quantizers[leaf] = self.train_vector_quantizer(leaf_data)

    def encode_vectors(self, vectors):
        encoded_vectors = []
        for vector in vectors:
            leaf = np.argmin([euclidean_distance(vector, centroid) for centroid in self.centroids])
            encoded = self.quantizers[leaf].predict([vector])[0]
            encoded_vectors.append((leaf, encoded))
        return encoded_vectors

    def search(self, query_vector, dataset, encoded_dataset, k):
        query_leaf = np.argmin([euclidean_distance(query_vector, centroid) for centroid in self.centroids])
        candidate_leaves = [query_leaf]  # In a real implementation, we'd find nearby leaves
        candidates = []
        for leaf in candidate_leaves:
            leaf_candidates = [i for i, (l, _) in enumerate(encoded_dataset) if l == leaf]
            candidates.extend(leaf_candidates)
        
        distances = []
        for idx in candidates:
            vector = dataset[idx]
            distance = euclidean_distance(query_vector, vector)
            distances.append((distance, idx))
        
        return heapq.nsmallest(k, distances)



########################################################################################
# HNSW
########################################################################################

"""
Algorithm:
2 parts: Navigable small work and skipped linked list.
General idea:
    There are multiple layers of graphs.
    Each layer is a navigable small world.
    The nearest discovered neighbor in current layer is used to jump to the next layer as the entrypoint to the next layer.
    We finish with the nearest discovered neighbor in the lowest layer.

    
Parameters post construction:
    Dict of dicts: {
        layer_index: {
            node_id: (vector, [list of neighbor node ids])
        }
    }
Search Algorithm:
    Start from the enter point. (Could be random node from the first layer)
    Search the layer for the nearest neighbor (use the entry points neighbors).
    Use the nearest neighbor to jump to the next layer and repeat search.
    Continue until the lowest layer is reached.
    Use the lowest layer's nearest neighbors as the result.

Construction Algorithm:
    Assign each node to a layer (few nodes on first layer, include all previous nodes on deeper layers).
    For each node on each layer, find a random set of nodes to connect to.
    Construct the dict of dicts.
"""


class HNSW:
    def __init__(self, num_layers=5, num_neighbors=3):
        self.num_layers = num_layers
        self.num_neighbors = num_neighbors
        self.graph = {
            layer_index: {
            #     {'id': None, 'vector': None, 'neighbors': [] }
            } for layer_index in range(num_layers)
        }

    def build(self, dataset):
        """
        The dataset is a list of vectors.
        """
        # Assign each node to layer 0
        for node_index, vector in enumerate(dataset):
            self.graph[0][node_index] = {
                'id': node_index,
                'vector': vector,
                'neighbors': []
            }
        # Assign each node to the rest of the layers
        for i in range(1, self.num_layers):
            nodes_in_previous_layer = list(self.graph[i-1].keys())
            num_nodes_to_select = max(1, int(0.2 * len(nodes_in_previous_layer)))  # Ensure at least 1 node is selected
            selected_nodes = np.random.choice(nodes_in_previous_layer, size=num_nodes_to_select, replace=False)
            for node_index in selected_nodes:
                vector = self.graph[0][node_index]['vector']
                self.graph[i][int(node_index)] = {
                    'id': node_index,
                    'vector': vector,
                    'neighbors': []
                }
        # Assign neighbors to each node
        for i in range(self.num_layers):
            current_nodes = list(self.graph[i].keys())
            for node_index, node in self.graph[i].items():
                node['neighbors'] = np.random.choice(current_nodes, size=max(1, min(self.num_neighbors, len(current_nodes))), replace=False)

        

    def search(self, query_vector):
        # Start from the enter point. (Could be random node from the first layer)
        # Search the layer for the nearest neighbor (use the entry points neighbors).
        # Use the nearest neighbor to jump to the next layer and repeat search.
        # Continue until the lowest layer is reached.
        # Use the lowest layer's nearest neighbors as the result.
        entry_node = np.random.choice(list(self.graph[self.num_layers-1].keys()))
        for i in range(self.num_layers-1,0,-1):
            entry_node = self._find_nearest_neighbor(query_vector, self.graph[i][entry_node]['neighbors'])
        return entry_node
    
    def _find_nearest_neighbor(self, query_vector, neighbors):
        nearest_neighbor = None
        min_distance = np.inf
        for neighbor in neighbors:
            distance = euclidean_distance(query_vector, self.graph[0][int(neighbor)]['vector'])
            if distance < min_distance:
                min_distance = distance
                nearest_neighbor = neighbor
        return nearest_neighbor




########################################################################################
# PQ Encoding
########################################################################################

class ProductQuantization:
    def __init__(self, M, K):
        self.M = M  # Number of subvectors
        self.K = K  # Number of centroids per subvector
        self.codebooks = []

    def train(self, dataset):
        dim = dataset.shape[1]
        subvector_size = dim // self.M
        for i in range(self.M):
            start = i * subvector_size
            end = (i + 1) * subvector_size if i < self.M - 1 else dim
            subvectors = dataset[:, start:end]
            kmeans = KMeans(n_clusters=self.K)
            kmeans.fit(subvectors)
            self.codebooks.append(kmeans.cluster_centers_)

    def encode_vectors(self, vectors):
        encoded_vectors = []
        for vector in vectors:
            encoded = []
            for i in range(self.M):
                start = i * (vector.shape[0] // self.M)
                end = (i + 1) * (vector.shape[0] // self.M) if i < self.M - 1 else vector.shape[0]
                subvector = vector[start:end]
                nearest_centroid_index = np.argmin([euclidean_distance(subvector, centroid) for centroid in self.codebooks[i]])
                encoded.append(nearest_centroid_index)
            encoded_vectors.append(encoded)
        return encoded_vectors

    def compute_asymmetric_distances(self, encoded_query, encoded_dataset):
        distances = []
        for encoded_vector in encoded_dataset:
            distance = 0
            for i in range(self.M):
                distance += euclidean_distance(self.codebooks[i][encoded_query[i]], self.codebooks[i][encoded_vector[i]])
            distances.append(distance)
        return distances

    def search(self, query_vector, dataset, encoded_dataset, k):
        encoded_query = self.encode_vectors([query_vector])[0]
        distances = self.compute_asymmetric_distances(encoded_query, encoded_dataset)
        return sorted(enumerate(distances), key=lambda x: x[1])[:k]










def test_algorithms():
    # Generate sample data
    np.random.seed(42)
    dataset = np.random.rand(1000, 128)
    query = np.random.rand(128)

    print("Testing ScaNN:")
    scann = ScaNN(num_leaves=10, num_clusters_per_block=8)
    scann.train(dataset)
    encoded_dataset = scann.encode_vectors(dataset)
    results = scann.search(query, dataset, encoded_dataset, k=5)
    print("Top 5 results (distance, index):", results)

    print("\nTesting HNSW:")
    hnsw = HNSW(num_layers=4, num_neighbors=10)
    hnsw.build(dataset)
    results = hnsw.search(query)
    print("Top 5 results (similarity, index):", results)

    print("\nTesting PQ Encoding:")
    pq = ProductQuantization(M=8, K=256)
    pq.train(dataset)
    encoded_dataset = pq.encode_vectors(dataset)
    results = pq.search(query, dataset, encoded_dataset, k=5)
    print("Top 5 results (index, distance):", results)

if __name__ == "__main__":
    test_algorithms()
