#!/usr/bin/env bash
# Interim public demo: local container + Cloudflare quick tunnel.
#
# Stopgap while Oracle Always Free capacity is hunted.  No account, no card:
# `cloudflared tunnel --url` issues a throwaway *.trycloudflare.com hostname.
# The hostname changes on every restart -- that is the deal with quick tunnels.
#
# Two flags below are load-bearing and must not be "simplified":
#
#   -v .../storage/pages/small:...:ro
#       Mount ONLY the thumbnails.  main.py:229 mounts the *parent* directory
#       (/pages -> storage/pages) with no auth and enumerable filenames, so
#       mounting storage/pages whole publishes storage/pages/full -- 3.8 GB of
#       scans of the textbook (RULE 3.1).  HARRISON_PAGE_FULL_RES=false does
#       NOT prevent this: it only changes which URL the app emits, not what
#       StaticFiles will serve.  The image ships an empty /app/storage/pages/full
#       so the path 404s.
#
#   -e FORWARDED_ALLOW_IPS=172.17.0.1
#       The docker bridge gateway, i.e. cloudflared on the host -- never "*".
#       uvicorn 0.49's _TrustedHosts.get_trusted_client_address() short-circuits
#       on always_trust and returns x_forwarded_for[0], the LEFTMOST entry,
#       which is whatever the visitor sent.  Naming the proxy makes it walk the
#       list in reverse to the rightmost untrusted hop instead.  Without this,
#       a spoofed X-Forwarded-For sets the key main.py:172 rate-limits on and
#       30 req/min becomes unlimited against the Gemini key pool.
set -euo pipefail

cd "$(dirname "$0")/.."
PORT=7860
NAME=harrison_demo

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" \
  --platform linux/amd64 \
  -p "127.0.0.1:${PORT}:7860" \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --memory 4g --memory-swap 4g \
  --pids-limit 256 \
  --restart unless-stopped \
  -e HARRISON_PAGE_FULL_RES=false \
  -e FORWARDED_ALLOW_IPS=172.17.0.1 \
  -v "$PWD/entrypoint.sh:/app/entrypoint.sh:ro" \
  -v "$PWD/artifacts/vectorstore:/app/artifacts/vectorstore:ro" \
  -v "$PWD/storage/pages/small:/app/storage/pages/small:ro" \
  -v "$PWD/backend/.env:/app/backend/.env:ro" \
  harrisongpt:latest >/dev/null

printf 'warming up'
for _ in $(seq 1 40); do
    curl -sf -o /dev/null -m 3 "http://127.0.0.1:${PORT}/health" && break
    printf '.'; sleep 10
done
echo

# Refuse to expose anything until the licensed full-res scans are proven absent.
# This is the check the whole mount layout exists for; if it ever passes 200,
# the tunnel must not open.
code=$(curl -s -o /dev/null -w '%{http_code}' \
    "http://127.0.0.1:${PORT}/pages/full/page_3074_full.png")
if [ "$code" != "404" ]; then
    echo "ABORT: /pages/full/ served HTTP $code, expected 404." >&2
    echo "       Full-resolution textbook scans would be public. Not opening a tunnel." >&2
    docker rm -f "$NAME" >/dev/null
    exit 1
fi
echo "ok: /pages/full/ returns 404, thumbnails only."

: >/tmp/harrison-tunnel.log
nohup caffeinate -i cloudflared tunnel --url "http://127.0.0.1:${PORT}" \
    --no-autoupdate >/tmp/harrison-tunnel.log 2>&1 &
for _ in $(seq 1 30); do
    url=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/harrison-tunnel.log | head -1)
    [ -n "$url" ] && { echo "public URL: $url"; exit 0; }
    sleep 3
done
echo "tunnel did not report a URL; see /tmp/harrison-tunnel.log" >&2
exit 1
