#!/usr/bin/env bash
# Assemble the exact set of files the private HF dataset needs, and nothing else.
#
# Exists because the obvious command is wrong and expensive:
#
#     hf upload <user>/harrisongpt-corpus . --repo-type=dataset
#
# That `.` means "this whole directory": data/ (649 MB), storage/ (4.2 GB),
# artifacts/ including the backup and staging trees, and .git.  Roughly 5 GB
# uploaded when 0.5 GB is needed, in a layout fetch_corpus.py does not expect.
#
# The staged tree mirrors the application's own paths, so scripts/fetch_corpus.py
# can snapshot_download straight into place with no post-processing.
#
# Usage:
#   ./scripts/stage_corpus.sh                 # stages to /tmp/harrisongpt-corpus
#   ./scripts/stage_corpus.sh /some/other/dir
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-/tmp/harrisongpt-corpus}"

mkdir -p "$DEST/artifacts/vectorstore" "$DEST/storage/pages"

for f in index.faiss chunks.json; do
    if [ ! -f "$SRC/artifacts/vectorstore/$f" ]; then
        echo "FATAL: $SRC/artifacts/vectorstore/$f is missing. Run 'git lfs pull'." >&2
        exit 1
    fi
    # A git-lfs pointer is ~130 bytes.  Uploading one produces a dataset that
    # looks fine and a Space that cannot load its index.
    size=$(wc -c < "$SRC/artifacts/vectorstore/$f")
    if [ "$size" -lt 1000000 ]; then
        echo "FATAL: $f is only ${size} bytes -- that is an lfs pointer, not the file." >&2
        echo "       Run 'git lfs pull' and try again." >&2
        exit 1
    fi
    cp "$SRC/artifacts/vectorstore/$f" "$DEST/artifacts/vectorstore/$f"
done

rsync -a --delete "$SRC/storage/pages/small/" "$DEST/storage/pages/small/"

# storage/pages/full is deliberately NOT staged: 3.8 GB that a free Space would
# re-pull on every wake.  HARRISON_PAGE_FULL_RES=false handles its absence.
if [ -e "$DEST/storage/pages/full" ]; then
    echo "FATAL: full-resolution renders staged. Remove them." >&2
    exit 1
fi
if [ -e "$DEST/data" ]; then
    echo "FATAL: data/ staged -- that is the raw textbook." >&2
    exit 1
fi

echo "Staged to $DEST"
du -sh "$DEST/artifacts/vectorstore" "$DEST/storage/pages/small" "$DEST"
echo
echo "Next (create it PRIVATE, and check flag names with 'hf upload --help'):"
echo "  hf upload <your-username>/harrisongpt-corpus \"$DEST\" . --repo-type=dataset --private"
