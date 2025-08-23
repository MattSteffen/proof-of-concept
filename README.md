# Applied Mathematics Projects

A hands-on learning repository for understanding machine learning algorithms, neural networks, and applied mathematics through implementation. This isn't about using existing libraries - it's about building algorithms from scratch to truly understand how they work.

Some ideas for this repo:
1. Have a cool idea
2. Implement an mvp
3. If it is cool enough, make it it's own folder
4. If it is still cool enough, rewrite it in another language (go or rust or c++)

## What This Is

This repository contains my implementations of various algorithms and techniques in machine learning, search, and clustering. The goal is to learn by doing - implementing algorithms from first principles rather than just calling library functions. Each implementation includes:

- **Naive versions**: Simple, educational implementations that prioritize clarity over performance
- **Benchmarks**: Performance comparisons between naive and optimized approaches
- **Analysis**: Complexity analysis, trade-offs, and when to use what
- **Real datasets**: Testing on actual data (IMDB reviews, Fashion MNIST, etc.)

## How I'm Using It

The approach is systematic:

1. **Implement the naive version** - Start with the simplest possible implementation to understand the core algorithm
2. **Benchmark performance** - Measure time, memory, and accuracy on real datasets
3. **Implement optimized versions** - Build more sophisticated versions using techniques like indexing, quantization, etc.
4. **Compare and analyze** - Understand the trade-offs between simplicity and performance
5. **Document insights** - Capture what I learned about when each approach is appropriate

This isn't about building production systems - it's about building intuition and understanding through implementation.

## Current Focus Areas

### Search Algorithms
- **Naive implementations**: Brute-force, KNN, K-means clustering
- **High-performance versions**: ScaNN, Faiss, HNSW, Product Quantization
- **Benchmarks**: Performance comparisons across different dataset sizes and dimensions

### Clustering Algorithms  
- **Simple**: Spectral clustering, K-means, Gaussian mixtures, OPTICS, HDBSCAN
- **Advanced**: PCA/UMAP + clustering, autoencoders + clustering, community detection algorithms

### Neural Networks
- **Basics**: Simple networks learning trigonometric functions
- **Diffusion models**: Text generation using diffusion processes
- **Graph Neural Networks**: Diffusion on graphs

### Machine Learning Projects
- **Text analysis**: IMDB sentiment analysis
- **Computer vision**: Fashion MNIST classification
- **Tabular data**: Titanic survival prediction, Iris classification

## TODO:

- [ ] General tasks:
  - [ ] Write benchmarks to compare them
  - [ ] Write up complexity analysis (storage, time)
  - [ ] Write up my own implementation
  - [ ] Code up open-source implementation and compare benchmarks
  - [ ] Use example set to make sure all implementations are correct
  - [ ] Write up use cases (when is it useful, what kind of data is it useful for)

## Search Algorithms

- [ ] Naive Search
  - [ ] Debug and confirm all naive search implementations
  - [ ] Algorithms: Brute-force, KNN, KMeans
- [ ] High Performance Search
  - [ ] ScaNN
  - [ ] Faiss
  - [ ] HNSW
  - [ ] PQ encoding
- [ ] Optimized search for huge datasets
  - [ ] PGVector
    - [ ] PGScale too
  - [ ] Faiss

## Clustering Algorithms

- [ ] Simple
  - [ ] Spectral clustering
  - [ ] KMeans clustering
  - [ ] Gaussian mixture models
  - [ ] OPTICS
  - [ ] HDBSCAN
  - [ ] Heirarchical clustering
- [ ] Advanced
  - [ ] PCA/UMAP + clustering
  - [ ] Autoencoders + clustering
  - [ ] Louvain community detection
  - [ ] Leiden algorithm
  - [ ] Girvan-Newman Algorithm
  - [ ] Fancy Heirarchical clustering
