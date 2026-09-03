# HarrisonGPT — Hugging Face Space (Docker SDK), free CPU basic tier.
#
# Build for the target architecture, not the build host:
#
#     docker build --platform linux/amd64 -t harrisongpt .
#
# A Mac mini M4 builds arm64 by default.  HF Spaces run x86_64, and torch and
# faiss-cpu ship different wheels per architecture, so an unqualified build
# produces an image that works on the desk and can still fail on the Space.
#
# Measured footprint of the running app (FAISS 16,983 vectors + BM25 corpus +
# BGE-M3 + cross-encoder): 1.47 GB peak RSS, 23 s warm-up.  Free CPU basic
# gives 16 GB, so the headroom is real.

FROM python:3.12-slim

# HF Spaces expects a non-root user at uid 1000.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONPATH=/app

WORKDIR /app

# torch comes from PyTorch's CPU index deliberately.  The default PyPI wheel for
# linux/amd64 bundles the CUDA runtime and is several GB larger, for a GPU this
# image will never have.  CODING_RULES 6.2: runtime target is CPU.
RUN pip install --no-cache-dir torch==2.12.1 \
        --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Hand both the home directory and /app to the runtime user before dropping
# privileges.  The pip steps above run as root with HOME=/home/user, which
# leaves a root-owned /home/user/.cache behind; without this chown the model
# bake below dies with EACCES on the HF cache, and fetch_corpus.py would later
# fail the same way writing into a root-owned /app.
RUN mkdir -p "$HF_HOME" \
        /app/artifacts/vectorstore \
        /app/storage/pages/small \
        /app/storage/pages/full \
    && chown -R user:user /home/user /app

USER user

# Bake both model weights into the image.  Free Spaces have ephemeral storage,
# so anything downloaded at runtime is re-downloaded on every wake -- that is
# 2.4 GB per cold start if these are not resident in a layer.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-m3'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY --chown=user backend/ /app/backend/
COPY --chown=user scripts/fetch_corpus.py /app/scripts/fetch_corpus.py
COPY --chown=user --chmod=755 entrypoint.sh /app/entrypoint.sh

EXPOSE 7860
ENTRYPOINT ["/app/entrypoint.sh"]
