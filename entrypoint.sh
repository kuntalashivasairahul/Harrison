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

# --proxy-headers is required; the allow-list is deliberately narrow.
#
# main.py builds every page-image URL from str(request.base_url).  Behind an
# HTTPS proxy, uvicorn only honours X-Forwarded-Proto from IPs in
# --forwarded-allow-ips, so without the proxy's address here base_url comes
# back as http:// on an https:// page, every thumbnail is blocked as mixed
# content, and the cited-pages rail renders as broken images.
#
# It must name the proxy, never "*".  uvicorn 0.49's
# _TrustedHosts.get_trusted_client_address() special-cases always_trust and
# returns x_forwarded_for[0] -- the LEFTMOST entry, which is whatever the
# visitor sent.  Every proxy appends, so a client that sends its own
# X-Forwarded-For controls the value main.py:172 keys the rate limiter on,
# and 30 req/min becomes unlimited against the Gemini key pool.  Naming the
# proxy instead makes uvicorn walk the list in reverse and return the
# rightmost untrusted hop: the real client, which cannot be forged from
# outside.
#
# Default is localhost, correct for a tunnel or a same-host reverse proxy.
# Override with FORWARDED_ALLOW_IPS when the proxy reaches this container
# from another address (e.g. a Docker bridge gateway).
exec uvicorn backend.api.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-7860}" \
    --proxy-headers \
    --forwarded-allow-ips="${FORWARDED_ALLOW_IPS:-127.0.0.1}"
