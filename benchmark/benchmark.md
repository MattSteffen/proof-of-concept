# Vector Search Algorithm Benchmark

This folder contains benchmarks for comparing various vector search algorithms, focusing on naive implementations such as brute-force and HNSW (Hierarchical Navigable Small World). The goal is to provide a standardized set of metrics for evaluating and comparing these algorithms.

## Table of Contents

1. [Metrics](#metrics)
2. [Benchmark Generation](#benchmark-generation)
3. [Interpreting Results](#interpreting-results)
4. [List of Benchmarks](#list-of-benchmarks)
5. [Comparison with Optimized Counterparts](#comparison-with-optimized-counterparts)

## Metrics

We use the following metrics to evaluate the performance of vector search algorithms:

### 1. Time Efficiency

- **Build Time**: Time taken to construct the index (where applicable).
- **Search Time**: Time taken to perform a single search query.
- **Throughput**: Number of queries processed per second.

### 2. Accuracy

- **Recall@k**: The proportion of true top-k nearest neighbors found.
- **Mean Reciprocal Rank (MRR)**: Average reciprocal of the rank of the true nearest neighbor.
- **Mean Average Precision (MAP)**: Mean of the average precision scores for each query.

### 3. Error Analysis

- **Mean Rank Error**: Average rank difference between found neighbors and true nearest neighbors.
- **Distance Ratio**: Ratio of distances between found neighbors and true nearest neighbors.
- **Worst-Case Error**: Maximum rank or distance error observed.

### 4. Data Distribution Sensitivity

- **Distribution-Controlled Accuracy**: Accuracy measures across different data distributions (e.g., uniform, Gaussian, clustered).
- **Intrinsic Dimensionality Impact**: How accuracy changes with the intrinsic dimensionality of the data.

### 5. Scalability

- **Memory Usage**: Peak memory consumption during index building and searching.
- **Index Size**: Size of the constructed index on disk.
- **Query Complexity**: How search time scales with dataset size and dimensionality.

## Benchmark Generation

To generate benchmarks:

1. Prepare diverse datasets:

   - Varying sizes (e.g., 10K, 100K, 1M vectors)
   - Different dimensionalities (e.g., 50D, 100D, 1000D)
   - Various distributions (uniform, Gaussian, real-world data)

2. For each algorithm:

   - Measure build time (if applicable)
   - Perform searches with a set of query vectors
   - Record all relevant metrics

3. Run multiple iterations to ensure statistical significance

4. Use the same hardware setup for fair comparisons

## Interpreting Results

- **Time Efficiency**: Lower build and search times are better. Higher throughput is better.
- **Accuracy**: Higher values for Recall@k, MRR, and MAP indicate better performance.
- **Error Analysis**: Lower values for Mean Rank Error and Distance Ratio are better. Pay attention to Worst-Case Error for reliability assessment.
- **Data Distribution Sensitivity**: Look for consistent performance across different distributions. Significant variations may indicate potential issues in certain scenarios.
- **Scalability**: Observe how metrics change as dataset size and dimensionality increase. Algorithms with sub-linear scaling are generally preferable for large-scale applications.

## List of Benchmarks

#### Naive Algorithms

- Brute-Force Vector Search
- KNN Search
- KMeans Search

#### High Performance Algorithms

- Faiss
- ScaNN
- HNSW
- Product Quantization

For each algorithm, we'll run benchmarks on:

- Different dataset sizes: 10K, 100K, 1M vectors
- Various dimensionalities: 50D, 100D, 1000D
- Multiple data distributions: Uniform, Gaussian, Clustered, Real-world datasets

## Comparison with Optimized Counterparts

To provide a comprehensive view, we'll also compare each naive implementation with its optimized counterpart:

1. Implement both naive and optimized versions of each algorithm.
2. Run the same set of benchmarks for both versions.
3. Calculate the performance gap:

   - Speedup factor: (Naive time / Optimized time)
   - Accuracy difference: (Optimized accuracy - Naive accuracy)
   - Memory efficiency: (Naive memory usage / Optimized memory usage)

4. Provide insights on:
   - Trade-offs between simplicity and performance
   - Scenarios where naive implementations might be sufficient
   - Importance of optimization for different use cases

By comparing naive implementations with their optimized counterparts and other algorithms, we can gain valuable insights into:

- The effectiveness of various optimization techniques
- The inherent strengths and weaknesses of each algorithm
- Suitable algorithms for different types of data and query patterns

This comprehensive benchmarking approach will help users make informed decisions when choosing and implementing vector search algorithms for their specific use cases.
