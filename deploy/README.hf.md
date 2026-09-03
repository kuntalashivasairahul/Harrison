---
title: HarrisonGPT
emoji: 🩺
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Medical RAG over Harrison's Principles of Internal Medicine
---

# HarrisonGPT

Retrieval-augmented question answering over *Harrison's Principles of Internal
Medicine*. Hybrid dense + lexical retrieval, cross-encoder reranking, and a
draft/verify LLM pass where no model is allowed to verify its own draft.

When the retrieved passages do not support an answer, it refuses, verbatim and
unsoftened, rather than producing a plausible one. That is the point of the
system.

## Notes for anyone reading this Space

- **First load after a pause is slow.** Free Spaces sleep after 48 h idle. A
  cold wake pulls the corpus and warms the encoder, FAISS index, BM25 corpus
  and reranker, which measured 23 s locally on top of the download.
- **The corpus is not in this repo.** The index and chunk text are derived from
  a copyrighted textbook and live in a private dataset that this Space reads
  with a scoped token. The code is public; the book is not.
- **Cited page images are thumbnails.** The full-resolution renders are 3.8 GB
  and are not deployed here, so the lightbox shows the WebP thumbnail.

## This file

This is the README for the **Space repo only**. Hugging Face parses the YAML
frontmatter above to configure the Space, and GitHub renders that same block as
a stray table, which is why the project README at the repo root does not carry
it. Copy this file to `README.md` inside the Space checkout; do not merge the
two.
