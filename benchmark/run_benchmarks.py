import numpy as np
from typing import List, Tuple
from benchmark.base import BenchmarkBase

def generate_dataset(num_vectors: int, vector_dim: int) -> np.ndarray:
    return np.random.rand(num_vectors, vector_dim)

def generate_queries(num_queries: int, vector_dim: int) -> np.ndarray:
    return np.random.rand(num_queries, vector_dim)

def compute_ground_truth(dataset: np.ndarray, queries: np.ndarray, k: int) -> List[List[int]]:
    ground_truth = []
    for query in queries:
        distances = np.linalg.norm(dataset - query, axis=1)
        ground_truth.append(np.argsort(distances)[:k].tolist())
    return ground_truth

class SearchBenchmark(BenchmarkBase):
    def __init__(self, name: str, algorithm):
        super().__init__(name)
        self.algorithm = algorithm

    def build(self, dataset: np.ndarray) -> None:
        self.algorithm.build(dataset)

    def search(self, query: np.ndarray, k: int) -> List[Tuple[float, int]]:
        return self.algorithm.search(query, k)

def main():
    # Generate dataset and queries
    num_vectors = 10000
    vector_dim = 128
    num_queries = 100
    k = 10

    dataset = generate_dataset(num_vectors, vector_dim)
    queries = generate_queries(num_queries, vector_dim)
    ground_truth = compute_ground_truth(dataset, queries, k)

    # Initialize algorithms
    # naive_faiss = NaiveFAISS(n_subvectors=8, n_clusters=256)
    # optimized_faiss = OptimizedFAISS(n_subvectors=8, n_clusters=256)
    # naive_hnsw = NaiveHNSW(num_neighbors=10)
    # optimized_hnsw = OptimizedHNSW(num_layers=4, num_neighbors=10)

    # Create benchmarks
    # benchmarks = [
    #     SearchBenchmark("Naive FAISS", naive_faiss),
    #     SearchBenchmark("Optimized FAISS", optimized_faiss),
    #     SearchBenchmark("Naive HNSW", naive_hnsw),
    #     SearchBenchmark("Optimized HNSW", optimized_hnsw),
    # ]

    # # Run benchmarks
    # for benchmark in benchmarks:
    #     benchmark.run_benchmark(dataset, queries, k, ground_truth)
    #     benchmark.print_results()
    #     benchmark.plot_results()

if __name__ == "__main__":
    main()