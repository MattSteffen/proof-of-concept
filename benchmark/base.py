from abc import ABC, abstractmethod
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Any

class BenchmarkBase(ABC):
    def __init__(self, name: str):
        self.name = name
        self.build_time = 0
        self.search_times = []
        self.accuracies = []

    @abstractmethod
    def build(self, dataset: np.ndarray) -> None:
        pass

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> List[Tuple[float, Any]]:
        pass

    def run_benchmark(self, dataset: np.ndarray, queries: np.ndarray, k: int, ground_truth: List[List[int]]) -> None:
        # Measure build time
        start_time = time.time()
        self.build(dataset)
        self.build_time = time.time() - start_time

        # Measure search time and accuracy
        for query, gt in zip(queries, ground_truth):
            start_time = time.time()
            results = self.search(query, k)
            self.search_times.append(time.time() - start_time)
            
            # Calculate accuracy
            retrieved_indices = [r[1] for r in results]
            accuracy = len(set(retrieved_indices) & set(gt[:k])) / k
            self.accuracies.append(accuracy)

    def plot_results(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot search times
        ax1.hist(self.search_times, bins=20)
        ax1.set_title(f"{self.name} - Search Time Distribution")
        ax1.set_xlabel("Search Time (s)")
        ax1.set_ylabel("Frequency")

        # Plot accuracies
        ax2.hist(self.accuracies, bins=20)
        ax2.set_title(f"{self.name} - Accuracy Distribution")
        ax2.set_xlabel("Accuracy")
        ax2.set_ylabel("Frequency")

        plt.tight_layout()
        plt.savefig(f"{self.name}_benchmark_results.png")
        plt.close()

    def print_results(self):
        print(f"=== {self.name} Benchmark Results ===")
        print(f"Build Time: {self.build_time:.4f} seconds")
        print(f"Average Search Time: {np.mean(self.search_times):.4f} seconds")
        print(f"Average Accuracy: {np.mean(self.accuracies):.4f}")
        print("====================================")