import numpy as np
import time

"""
Each search function will be a class that has the following methods:
- init: takes in the data
- train: trains the model
- search: takes in a query and returns the closest vectors

Benchmarks will be doing:
    Accuracy
    Time taken
"""
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple

# Import the search classes (assuming they're in a file named vector_search_classes.py)
from search.naive_search import BruteForceSearch, SimpleFAISS, KNNSearch, KMeansSearch

def run_search(searcher, data: np.ndarray, query: np.ndarray, k: int = 1) -> Tuple[List[Tuple[float, np.ndarray]], float, float]:
    start_train = time.time()
    searcher.train(data)
    end_train = time.time()
    train_time = end_train - start_train

    start_search = time.time()
    result = searcher.search(query, k)
    end_search = time.time()
    search_time = end_search - start_search

    return result, train_time, search_time

# Create search functions
search_functions = [
    ('BruteForceSearch', lambda: BruteForceSearch()),
    ('SimpleFAISS', lambda: SimpleFAISS(n_subvectors=8, n_clusters=90)),
    ('KNNSearch', lambda: KNNSearch()),
    ('KMeansSearch', lambda: KMeansSearch(k_clusters=10))
]

# TODO: Create various sizes and various dimensions not just 768
# Create 3 np arrays of various sizes
sizes = [100, 1000, 10000]  # Small, Medium, Large
data_arrays = [np.random.rand(size, 768) for size in sizes]  # Assuming 768-dimensional vectors
queries = [np.random.rand(768) for _ in sizes]

def run_benchmarks():
    results = []

    for i, (data, query) in enumerate(zip(data_arrays, queries)):
        print(f"\nRunning searches on dataset of size {sizes[i]}")
        
        # Get the correct answer using brute force search
        bf_searcher = BruteForceSearch()
        correct_answer, _, _ = run_search(bf_searcher, data, query)
        correct_answer = correct_answer[0]
        
        for search_name, search_constructor in search_functions:
            searcher = search_constructor()
            result, train_time, search_time = run_search(searcher, data, query)
            result = result[0]
            
            # Check if the result is accurate (within a small tolerance)
            is_accurate = np.allclose(result[1], correct_answer[1], atol=1e-6)
            
            results.append({
                'Size': sizes[i],
                'Algorithm': search_name,
                'Train Time': train_time,
                'Search Time': search_time,
                'Total Time': train_time + search_time,
                'Accurate': is_accurate
            })
            
            print(f"Function {search_name}: Train time: {train_time:.6f} s, Search time: {search_time:.6f} s, Accurate: {is_accurate}")

    return pd.DataFrame(results)

def plot_results(df):
    plt.figure(figsize=(15, 10))
    
    # Plot training time
    plt.subplot(2, 1, 1)
    for algo in df['Algorithm'].unique():
        data = df[df['Algorithm'] == algo]
        plt.plot(data['Size'], data['Train Time'], marker='o', label=algo)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Dataset Size')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time vs Dataset Size')
    plt.legend()
    plt.grid(True)

    # Plot search time
    plt.subplot(2, 1, 2)
    for algo in df['Algorithm'].unique():
        data = df[df['Algorithm'] == algo]
        plt.plot(data['Size'], data['Search Time'], marker='o', label=algo)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Dataset Size')
    plt.ylabel('Search Time (s)')
    plt.title('Search Time vs Dataset Size')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    plt.close()

if __name__ == "__main__":
    results_df = run_benchmarks()
    
    print("\nBenchmark Results:")
    print(results_df.to_string(index=False))
    
    results_df.to_csv('benchmark_results.csv', index=False)
    print("\nResults saved to 'benchmark_results.csv'")
    
    plot_results(results_df)
    print("Plots saved to 'benchmark_results.png'")