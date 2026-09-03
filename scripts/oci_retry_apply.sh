#!/usr/bin/env bash
# Retry an OCI Resource Manager apply until ARM capacity appears.
#
# Oracle's Always Free Ampere (A1) shapes are chronically exhausted in popular
# regions.  A correct configuration fails with:
#
#     Error: 500-InternalError, Out of host capacity.
#
# That is not a configuration problem and no amount of editing fixes it; the
# only remedy is to keep asking until a host frees up, which can take hours.
# Clicking Apply by hand gets you throttled ("Too many requests") long before
# it gets you an instance, because rapid retries do not make capacity appear.
#
# This paces the attempts, distinguishes the errors worth retrying from the
# ones that are not, and stops the moment it succeeds.
#
# Usage:
#   ./scripts/oci_retry_apply.sh <stack-ocid> [interval-seconds]
#
# Find the stack OCID:  Console > Developer Services > Resource Manager >
# Stacks > harrisongpt > Stack information > OCID.
set -uo pipefail

STACK_ID="${1:-}"
INTERVAL="${2:-180}"       # 3 min. Below ~120s Oracle starts throttling.
MAX_HOURS="${MAX_HOURS:-12}"

if [ -z "$STACK_ID" ]; then
    echo "usage: $0 <stack-ocid> [interval-seconds]" >&2
    exit 64
fi
if ! command -v oci >/dev/null 2>&1; then
    echo "FATAL: the 'oci' CLI is not installed or not on PATH." >&2
    echo "       brew install oci-cli   then   oci setup config" >&2
    exit 1
fi

deadline=$(( $(date +%s) + MAX_HOURS * 3600 ))
attempt=0

log() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

log "Retrying stack apply every ${INTERVAL}s for up to ${MAX_HOURS}h."
log "Stack: $STACK_ID"
log "Ctrl-C to stop. Progress is also visible in the console under Jobs."

while [ "$(date +%s)" -lt "$deadline" ]; do
    attempt=$(( attempt + 1 ))

    job_id=$(oci resource-manager job create-apply-job \
                --stack-id "$STACK_ID" \
                --execution-plan-strategy AUTO_APPROVED \
                --query 'data.id' --raw-output 2>/dev/null)

    if [ -z "$job_id" ]; then
        # Usually the 429 throttle. Back off harder than the normal interval:
        # hammering through a throttle only extends it.
        log "attempt ${attempt}: could not create job (likely throttled). Backing off 300s."
        sleep 300
        continue
    fi

    # Poll this job to completion before deciding anything.
    while :; do
        state=$(oci resource-manager job get --job-id "$job_id" \
                    --query 'data."lifecycle-state"' --raw-output 2>/dev/null)
        case "$state" in
            SUCCEEDED|FAILED|CANCELED) break ;;
            *) sleep 10 ;;
        esac
    done

    if [ "$state" = "SUCCEEDED" ]; then
        log "attempt ${attempt}: SUCCEEDED — the instance is provisioning."
        compartment=$(oci resource-manager stack get --stack-id "$STACK_ID" \
                        --query 'data."compartment-id"' --raw-output 2>/dev/null)
        echo
        oci compute instance list --compartment-id "$compartment" \
            --display-name harrisongpt \
            --query 'data[0].{name:"display-name",state:"lifecycle-state",id:id}' \
            --output table 2>/dev/null
        echo
        log "Get the public IP with:"
        log "  oci compute instance list-vnics --instance-id <instance-ocid> \\"
        log "      --query 'data[0].\"public-ip\"' --raw-output"
        exit 0
    fi

    # Failed. Retry only if it is the capacity error; anything else is a real
    # problem and looping on it just wastes hours.
    logs=$(oci resource-manager job get-job-logs --job-id "$job_id" 2>/dev/null \
           | grep -iE "out of host capacity|429|TooManyRequests" | head -3)

    if echo "$logs" | grep -qi "out of host capacity"; then
        log "attempt ${attempt}: out of capacity. Retrying in ${INTERVAL}s."
    elif echo "$logs" | grep -qiE "429|TooManyRequests"; then
        log "attempt ${attempt}: throttled. Backing off 300s."
        sleep 300
        continue
    else
        log "attempt ${attempt}: FAILED for a reason that is not capacity. Stopping."
        echo "--- job $job_id ---" >&2
        oci resource-manager job get-job-logs --job-id "$job_id" 2>/dev/null | tail -25 >&2
        exit 1
    fi

    sleep "$INTERVAL"
done

log "Gave up after ${MAX_HOURS}h and ${attempt} attempts. Capacity never appeared."
log "Try again overnight, or early morning IST when contention is lowest."
exit 2
