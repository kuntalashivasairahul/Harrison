# backend/agents/__init__.py
"""
HarrisonGPT — Pre-retrieval and post-retrieval agentic layer.

Agents intercept or augment the RAG pipeline without touching retrieval
math or FastAPI routing (CODING_RULES.md §2.1).

Modules
-------
query_optimizer : Expands and disambiguates raw user queries before retrieval.
"""
