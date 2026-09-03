# The corpus: where it lives and how to get it back

The corpus is **not in this repository** and must never be put back into it.
This document is the retrieval path — read it before a deploy, a fresh clone,
or any attempt to rebuild the index.

## Why it is not in git

`data/harrison.md` is the complete text of *Harrison's Principles of Internal
Medicine*: 4,179,980 words, 30 MB, 409 chapters. `chunks.json` is the same
prose re-chunked with page numbers attached. Both are licensed content under
`CODING_RULES.md` RULE 3.1, so neither may sit in a public repo or a public
image layer.

They *were* committed, and the repository *was* public, from 2026-03-05 until
2026-09-03. History was rewritten on 2026-09-03 to remove them. The two-repo
split described in `CLAUDE.md` was always the correct design; it guarded the
Hugging Face Space door while this went out the GitHub one.

**git-lfs is no longer a prerequisite.** Nothing in this repo is LFS-tracked
any more. `CLAUDE.md`'s runtime section still says otherwise in older copies —
this file is authoritative.

## What was removed

Seven paths, stripped from every branch and from all history:

| path | bytes | sha256 |
|---|---|---|
| `data/harrison.md` | 30,158,309 | `3ad65f81c163d3fff1160a4f042c51544464e4dffec8b338f03184a86edbc1bc` |
| `artifacts/vectorstore/chunks.json` | 33,403,008 | `a384de30d6dc1d0427d2cf0012d32cca496964e75b2efc0774a9216447c307b5` |
| `artifacts/vectorstore/index.faiss` | 69,562,413 | `c6968f8b1ea958fad8dc08ad02c879d244c788091da33491d6df9fc0a9ad8d30` |
| `artifacts/vectorstore_backup/20260613T084900Z/chunks.json` | 32,484,794 | `20905377085f913226d083889f39d346f5a443bd76d145d7f12ab41c58462b32` |
| `artifacts/vectorstore_staging/table_chunks.json` | 33,403,008 | `a384de30d6dc1d0427d2cf0012d32cca496964e75b2efc0774a9216447c307b5` |
| `backend/vectorstore/chunks.json` | 32,484,794 | (deleted from HEAD 2026-03-07, still in history until the rewrite) |
| `artifacts/semantic_cache.json` | 216,687 | (runtime state, deleted from HEAD 2026-08-20) |

`vectorstore_staging/table_chunks.json` is byte-identical to
`vectorstore/chunks.json` — same sha256, not a distinct artifact.

Deliberately **kept**: `artifacts/vectorstore/README.md`,
`storage/pages/{full,small}/.gitkeep`.

## Copy 1 — the local archive (authoritative today)

```
~/Harrison_corpus_archive/
├── SHA256SUMS
├── data/harrison.md
└── artifacts/
    ├── vectorstore/{chunks.json,index.faiss}
    ├── vectorstore_backup/20260613T084900Z/chunks.json
    └── vectorstore_staging/table_chunks.json
```

Verify it at any time:

```bash
cd ~/Harrison_corpus_archive && shasum -a 256 -c SHA256SUMS   # expect 5x OK
```

Those checksums were verified against git's own LFS object IDs before the
history rewrite, so they are known-good copies of what the repo used to track.

Restore into a working tree:

```bash
cd ~/Developer/AI_Projects/nlp_models/Harrison
mkdir -p data artifacts/vectorstore
cp ~/Harrison_corpus_archive/data/harrison.md data/
cp ~/Harrison_corpus_archive/artifacts/vectorstore/{chunks.json,index.faiss} artifacts/vectorstore/
```

`.gitignore` covers all of these, so a restore cannot re-commit them.

> This archive lives on one disk. It is not a backup until it exists somewhere
> else too — an external drive or a private cloud folder. Do that.

## Copy 2 — the private Hugging Face dataset (the deploy path)

This is how a server gets the corpus. It is what Oracle will use.

**Publish** (from a machine that has the corpus):

```bash
./scripts/stage_corpus.sh                 # stages ~549 MB to /tmp/harrisongpt-corpus
hf upload <user>/harrisongpt-corpus /tmp/harrisongpt-corpus --repo-type=dataset --private
```

`stage_corpus.sh` assembles exactly `index.faiss`, `chunks.json` and the WebP
thumbnails, and refuses to stage a git-lfs pointer, `data/`, or the 3.8 GB
full-res renders. Do not `hf upload .` — that ships ~5 GB in a layout
`fetch_corpus.py` does not expect.

**Retrieve** — automatic. `entrypoint.sh` runs `scripts/fetch_corpus.py` before
uvicorn, which pulls only:

```
artifacts/vectorstore/index.faiss
artifacts/vectorstore/chunks.json
storage/pages/small/*
```

It needs two environment variables in `backend/.env`:

| var | purpose |
|---|---|
| `HARRISON_CORPUS_REPO` | the private dataset id |
| `HF_TOKEN` | read-scoped token for it |

It is a no-op when the files are already on disk, so it costs nothing locally.
The dataset **must stay private** — it holds the same licensed content this
repo was just cleaned of.

## Copy 3 — rebuild from source

If both copies are lost, the corpus is reproducible from the source PDF:

| step | script | output |
|---|---|---|
| 1. PDF → page images | `scripts/convert_pdf.py` | `storage/pages/{full,small}/` |
| 2. PDF → markdown | (external conversion) | `data/harrison.md` |
| 3. markdown → index | `scripts/ingest_tables_aware.py` | `artifacts/vectorstore_staging/` |

Step 3 writes to `vectorstore_staging/`; promote to `artifacts/vectorstore/`
once verified. Expect this to take hours and to produce a **different**
`index.faiss` (embedding non-determinism), so the checksums above will not
match a rebuild. `chunks.json` should be stable if the chunker is unchanged.

## What a deploy actually needs

Minimum for the app to start healthy (`/health` returns `status: "ok"`):

- `artifacts/vectorstore/index.faiss` — 16,983 vectors, dim 1024
- `artifacts/vectorstore/chunks.json` — the chunk registry
- `storage/pages/small/*.webp` — 450 MB of thumbnails, for the cited-pages rail

Not needed, and must not be deployed publicly:

- `storage/pages/full/*.png` — 3.8 GB. Set `HARRISON_PAGE_FULL_RES=false` so
  `resolve_page_urls()` points `full_url` at the thumbnail instead of a 404.
  **That flag is not a security control.** `/pages` is an unauthenticated
  StaticFiles mount of the *parent* directory with enumerable filenames, so if
  `storage/pages/full/` exists in the container it is public regardless of the
  flag. Mount only `storage/pages/small`, as `scripts/demo_tunnel.sh` does.
- `data/harrison.md` — source text, never needed at runtime.
