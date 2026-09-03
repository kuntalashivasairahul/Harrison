#!/usr/bin/env sh
# Boot order matters here, and it is not arbitrary.
#
# backend/api/main.py mounts StaticFiles("storage/pages") at import time, and
# StaticFiles raises RuntimeError when the directory does not exist.  So the
# corpus fetch cannot live in the FastAPI lifespan handler: by the time
# lifespan runs, the mount has already failed and the container is dead.  It
# also cannot live anywhere under backend/, because CLAUDE.md's import-time
# purity rule forbids networked work at import.  This script is the only
# correct home for it.
set -eu

python /app/scripts/fetch_corpus.py

# --proxy-headers with a wildcard allow-list is deliberate, not sloppy.
# main.py builds every page-image URL from str(request.base_url).  Behind HF's
# HTTPS proxy, uvicorn only honours X-Forwarded-Proto from IPs in
# --forwarded-allow-ips, which defaults to 127.0.0.1.  HF's proxy is not
# localhost from inside this container, so without the wildcard base_url comes
# back as http:// on an https:// page, every thumbnail is blocked as mixed
# content, and the cited-pages rail renders as broken images.
#
# The wildcard is safe *here* specifically because nothing but the platform's
# own proxy can reach this container -- there is no untrusted client in a
# position to spoof the header.  Running this image anywhere with direct
# public ingress requires narrowing it to that proxy's address.
exec uvicorn backend.api.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-7860}" \
    --proxy-headers \
    --forwarded-allow-ips="*"
