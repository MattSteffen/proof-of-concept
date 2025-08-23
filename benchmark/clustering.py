from benchmark.base import BenchmarkBase
import numpy as np
from typing import List, Tuple

class ClusteringBenchmark(BenchmarkBase):
    def __init__(self, name: str, algorithm):
        super().__init__(name)
        self.algorithm = algorithm

    def build(self, dataset: np.ndarray) -> None:
        self.algorithm.train(dataset)

    def search(self, query: np.ndarray, k: int) -> List[Tuple[float, int]]:
        return self.algorithm.search(query, k)