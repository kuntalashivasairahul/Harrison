# Production Vectorstore Snapshot

This directory contains the runnable FAISS retrieval snapshot required by the
API. Keep `index.faiss` and `chunks.json` together; they are produced from the
tracked `data/harrison.md` source using `scripts/ingest_tables_aware.py`.

Expected snapshot:

| File | SHA-256 | Purpose |
| --- | --- | --- |
| `index.faiss` | `c6968f8b1ea958fad8dc08ad02c879d244c788091da33491d6df9fc0a9ad8d30` | 16,983 1024-dimensional embeddings |
| `chunks.json` | `a384de30d6dc1d0427d2cf0012d32cca496964e75b2efc0774a9216447c307b5` | Matching chunk text and page metadata |

To regenerate deliberately, install dependencies and run:

```bash
.venv312/bin/python scripts/ingest_tables_aware.py --promote
```

On Windows use `.venv312\Scripts\python.exe` instead. Regeneration downloads
the embedding model on first use and overwrites this snapshot only with
`--promote`; it is not needed for a normal clone-and-run workflow.
