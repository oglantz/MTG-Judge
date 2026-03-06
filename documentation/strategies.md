# Retrieval Strategy Summary

## Current Pipeline
- Parse the MTG rules text into structured documents (one document per rules chunk).
- Generate dense embeddings with `BAAI/bge-base-en-v1.5` (normalized).
- Index embeddings in FAISS using inner-product similarity (`IndexFlatIP`, 768 dims).
- Retrieve with a dense vector retriever (`top_k=30` candidates).
- Retrieve with a BM25 sparse retriever (`top_k=75` candidates).
- Fuse dense + BM25 results with Reciprocal Rank Fusion (`top_k=50`).
- Rerank with a sentence-transformer cross-encoder (`BAAI/bge-reranker-base`, `top_n=10` final).
- Print the top reranked nodes for inspection and evaluation.

## Rationale
- Dense embeddings capture semantic similarity beyond exact wording.
- Normalization makes inner-product scoring behave like cosine similarity.
- FAISS provides fast, scalable similarity search on the embedding vectors.
- BM25 captures exact term matches and rare rule text tokens.
- RRF fusion boosts items that score well in both systems.
- Cross-encoder reranking improves final relevance quality at the cost of speed.

## Current Defaults
- Example query used in testing: “Can you attack with a tapped creature?”
- Dense `top_k`: 30
- BM25 `top_k`: 75
- Fusion `top_k`: 50
- Reranker `top_n`: 10

## Future Strategy Ideas
- Tune `top_k` values for speed/quality tradeoffs.
- Evaluate alternative embedding models (larger or domain-tuned).
- Add query rewriting for MTG rules-specific phrasing.
- Persist the index to disk to avoid rebuilding on every run.
