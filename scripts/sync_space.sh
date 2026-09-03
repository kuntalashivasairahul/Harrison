#!/usr/bin/env bash
# Assemble a Hugging Face Space checkout from this repo, copying ONLY the files
# that are safe to publish.
#
# Why this exists, and why you must not just add the Space as a git remote:
#
#   $ git ls-files | grep -E '^(artifacts|data)/'
#   artifacts/vectorstore/chunks.json       <- 33 MB of verbatim Harrison's text
#   artifacts/vectorstore/index.faiss
#   artifacts/vectorstore_backup/.../chunks.json
#   artifacts/vectorstore_staging/table_chunks.json
#   data/harrison.md                        <- 30 MB, the whole book
#
# Those are tracked, and they are in the commit history.  `git push space main`
# would publish the textbook to a public Space, history included, and a git
# push is not something you can take back once it has been fetched or indexed.
# .dockerignore does not help here: it governs the image, not the git push.
#
# So the Space gets its own clean history containing only deploy files.
#
# Usage:
#   git clone https://huggingface.co/spaces/<you>/harrisongpt ~/hf-harrisongpt
#   ./scripts/sync_space.sh ~/hf-harrisongpt
#   cd ~/hf-harrisongpt && git add -A && git commit -m "deploy" && git push
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-}"

if [ -z "$DEST" ]; then
    echo "usage: $0 /path/to/space-checkout" >&2
    exit 64
fi
if [ ! -d "$DEST/.git" ]; then
    echo "FATAL: $DEST is not a git checkout. Clone the Space repo there first." >&2
    exit 1
fi
if [ "$(cd "$DEST" && pwd)" = "$SRC" ]; then
    echo "FATAL: destination is this repo. That would defeat the entire point." >&2
    exit 1
fi

echo "Syncing deploy files -> $DEST"

# Allowlist, not denylist.  A denylist silently ships whatever it forgot.
rsync -a --delete \
    --exclude '__pycache__/' --exclude '*.pyc' \
    --exclude '.env' --exclude '.impeccable/' \
    "$SRC/backend/" "$DEST/backend/"

mkdir -p "$DEST/scripts"
cp "$SRC/scripts/fetch_corpus.py" "$DEST/scripts/fetch_corpus.py"
cp "$SRC/Dockerfile"             "$DEST/Dockerfile"
cp "$SRC/entrypoint.sh"          "$DEST/entrypoint.sh"
cp "$SRC/.dockerignore"          "$DEST/.dockerignore"
cp "$SRC/deploy/README.hf.md"    "$DEST/README.md"   # HF reads the YAML frontmatter

# ---------------------------------------------------------------------------
# Guard.  Everything above is intent; this is verification.  It runs against
# what actually landed on disk, because the failure mode here is irreversible.
# ---------------------------------------------------------------------------
fail=0

for forbidden in data storage artifacts; do
    if [ -e "$DEST/$forbidden" ]; then
        echo "FATAL: $forbidden/ present in the Space checkout." >&2
        fail=1
    fi
done

if [ -e "$DEST/backend/.env" ]; then
    echo "FATAL: backend/.env present in the Space checkout." >&2
    fail=1
fi

# Nothing legitimate in a code-only deploy is large.  The vendored three.js is
# the biggest file at well under a megabyte, so anything above this threshold
# means corpus content slipped through a path nobody thought of.
big="$(find "$DEST" -path "$DEST/.git" -prune -o -type f -size +5M -print 2>/dev/null || true)"
if [ -n "$big" ]; then
    echo "FATAL: files over 5 MB found. A code-only deploy has none:" >&2
    echo "$big" >&2
    fail=1
fi

if [ "$fail" -ne 0 ]; then
    echo >&2
    echo "Nothing was committed. Fix the above before pushing." >&2
    exit 1
fi

echo "OK: $(find "$DEST" -path "$DEST/.git" -prune -o -type f -print | wc -l | tr -d ' ') files, none licensed, none over 5 MB."
echo
echo "Next:"
echo "  cd $DEST"
echo "  git add -A && git commit -m 'deploy' && git push"
