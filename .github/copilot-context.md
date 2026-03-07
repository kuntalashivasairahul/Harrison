# HarrisonGPT Architecture Context

Project name: HarrisonGPT

## Stack
- FastAPI
- Retrieval Augmented Generation (RAG)
- FAISS vector search
- BM25 lexical retrieval
- Cross-encoder reranking
- Groq LLaMA models

## Pipeline Flow
Query
-> Query Expansion
-> Hybrid Retrieval (FAISS + BM25)
-> Reciprocal Rank Fusion
-> Neighbor Chunk Expansion
-> Cross-Encoder Reranking
-> Context Fusion
-> Evidence Extraction
-> LLM Generation
-> Verification
-> Final Answer

## Key Modules
- backend/api/main.py
- backend/retrieval/rag.py
- backend/retrieval/rerank.py
- backend/retrieval/embeddings.py
- backend/utils/fusion.py
- backend/processing/evidence.py
- backend/llm/llm.py
