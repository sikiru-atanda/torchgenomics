#!/usr/bin/env bash
# Shared memory + disk pre-flight check for Pillar B external-tool harnesses.
#
# Per spec section 5.3 (the user's hard rule): before each fetch_data.sh
# and each run_<tool>.sh, assert sufficient disk and memory headroom.
# If either check fails, abort with explicit message — do not download
# or run partially.
#
# Usage:
#     source "$(dirname "$0")/../_lib/preflight.sh"
#     preflight_check <tool_name> <required_disk_gb> <required_ram_gb>
#
# Example:
#     source "$(dirname "$0")/../_lib/preflight.sh"
#     preflight_check "ldsc" 10 4

set -euo pipefail

# Print to stderr.
err() { printf '%s\n' "$*" >&2; }

# Returns the human-readable disk-available figure on /home in GB (integer).
_disk_avail_gb_home() {
    df --output=avail -BG /home 2>/dev/null | tail -1 | tr -d 'G '
}

# Returns the available memory in GB (integer, from `free -g` "available" col 7).
_ram_avail_gb() {
    free -g | awk '/^Mem:/ {print $7}'
}

# Returns currently free swap in GB.
_swap_free_gb() {
    free -g | awk '/^Swap:/ {print $4}'
}

# preflight_check <tool> <required_disk_gb> <required_ram_gb>
preflight_check() {
    local tool="${1:?usage: preflight_check <tool> <required_disk_gb> <required_ram_gb>}"
    local req_disk_gb="${2:?missing required disk gb}"
    local req_ram_gb="${3:?missing required ram gb}"

    local avail_disk; avail_disk=$(_disk_avail_gb_home)
    local avail_ram; avail_ram=$(_ram_avail_gb)
    local free_swap; free_swap=$(_swap_free_gb)

    err ""
    err "[preflight: ${tool}]"
    err "  required: ${req_disk_gb} GB disk on /home, ${req_ram_gb} GB RAM"
    err "  observed: ${avail_disk} GB disk avail, ${avail_ram} GB RAM avail, ${free_swap} GB swap free"

    if (( avail_disk < req_disk_gb )); then
        err ""
        err "ABORT: insufficient disk on /home"
        err "  needed: ${req_disk_gb} GB; available: ${avail_disk} GB"
        err "Free up space before retrying."
        exit 1
    fi

    if (( avail_ram < req_ram_gb )); then
        err ""
        err "ABORT: insufficient available RAM"
        err "  needed: ${req_ram_gb} GB; available: ${avail_ram} GB"
        err "Free up memory or wait for other processes to finish."
        exit 1
    fi

    err "  OK"
    err ""
}

# preflight_check_with_data_size <tool> <data_download_gb> <peak_ram_gb>
# Convenience wrapper that adds standard headroom multipliers per the spec
# section 5.3 contract:
#   - disk: download_size * 2 + working_set * 2  (estimated as download * 4)
#   - RAM:  peak_working_set * 1.5 + 4 GB headroom
preflight_check_with_data_size() {
    local tool="${1:?usage: preflight_check_with_data_size <tool> <data_gb> <peak_ram_gb>}"
    local data_gb="${2:?missing data gb}"
    local peak_ram_gb="${3:?missing peak ram gb}"

    local req_disk_gb=$(( data_gb * 4 ))
    local req_ram_gb=$(( (peak_ram_gb * 3 + 1) / 2 + 4 ))

    preflight_check "${tool}" "${req_disk_gb}" "${req_ram_gb}"
}

# Verify a downloaded file's checksum.
# verify_checksum <file> <expected_sha256>
verify_checksum() {
    local file="${1:?usage: verify_checksum <file> <expected_sha256>}"
    local expected="${2:?missing expected sha256}"

    if [[ ! -f "${file}" ]]; then
        err "ABORT: file not found for checksum verification: ${file}"
        exit 1
    fi

    local actual; actual=$(sha256sum "${file}" | awk '{print $1}')
    if [[ "${actual}" != "${expected}" ]]; then
        err "ABORT: checksum mismatch on ${file}"
        err "  expected: ${expected}"
        err "  actual:   ${actual}"
        err "Re-download or update the expected checksum."
        exit 1
    fi
    err "[checksum OK] ${file}"
}
