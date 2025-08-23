# Better Semantic Search

RAG is pretty popular right now. Sometimes you want to comb through results with a certain persepecive, like looking for red flags as a financial anaylst or as a legal analyst. Semantic embeddings treat all perspectives the same. What if you have several different methods of indexing a set of documents and dynamically search for the correct sources with the correct perspectives.

I propose a bipartite graph as the index for embedded sources. Embed the source via multiple models with different clustering heads, build a learnable bipartite graph that allows you to traverse the graph looking on the right planes (with the right perspectives) for the correct documents.

To train your own embedding model for differnent perspectives, I propose a siamese network bulding off of the embedding model `all-minilm`.

How this can be shown effective is:

- Reranking accuracy?
