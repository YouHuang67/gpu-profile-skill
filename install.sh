#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKILLS_SRC="${SCRIPT_DIR}/skills"
SKILLS_DST=""
TARGET="claude"

SKILL_DIRS=(ncu-profile nsys-profile profile)

usage() {
    cat <<EOF
GPU Profile Skill installer. Works with Claude Code and Codex.

Installs three slash commands for GPU kernel profiling:
  /ncu-profile    NCU hardware counter analysis
  /nsys-profile   NSYS timeline analysis
  /profile        Combined NCU + NSYS

Usage:
  $(basename "$0") <command> [-t <target>]

Commands:
  install       Install the skills (creates symlinks)
  uninstall     Remove the skills (deletes symlinks)
  status        Show what is installed
  check         Check GPU, CUDA, NCU, NSYS, sudo, Python

Targets (-t):
  claude        ~/.claude/skills/      (default, Claude Code)
  codex         .agents/skills/        (Codex, in current directory)

Examples:
  # First-time setup:
  $(basename "$0") check                    # verify prerequisites
  $(basename "$0") install                  # install to ~/.claude/skills/
  $(basename "$0") status                   # confirm installed

  # Codex:
  $(basename "$0") install -t codex         # install to ./.agents/skills/
  $(basename "$0") uninstall -t codex       # remove from ./.agents/skills/

  # Clean up:
  $(basename "$0") uninstall                # remove from ~/.claude/skills/
EOF
}

_resolve_target() {
    case "${TARGET}" in
        claude)  SKILLS_DST="${HOME}/.claude/skills" ;;
        codex)   SKILLS_DST="$(pwd)/.agents/skills" ;;
        *) die "Unknown target: ${TARGET}. Use: claude, codex" ;;
    esac
}

info()  { echo "[INFO] $*"; }
warn()  { echo "[WARN] $*" >&2; }
die()   { echo "[ERROR] $*" >&2; exit 1; }

do_install() {
    mkdir -p "${SKILLS_DST}"

    local installed=0
    local skipped=0
    local cleaned=0

    # Clean up stale symlinks that point into this repo but are no longer in SKILL_DIRS
    for dst in "${SKILLS_DST}"/*; do
        if [ -L "${dst}" ]; then
            local target
            target="$(readlink "${dst}")"
            if [[ "${target}" == "${SKILLS_SRC}/"* ]]; then
                local name
                name="$(basename "${dst}")"
                if [[ ! " ${SKILL_DIRS[*]} " =~ " ${name} " ]]; then
                    warn "Removing stale symlink: ${dst}"
                    rm "${dst}"
                    cleaned=$((cleaned + 1))
                fi
            fi
        fi
    done

    for dir in "${SKILL_DIRS[@]}"; do
        local src="${SKILLS_SRC}/${dir}"
        local dst="${SKILLS_DST}/${dir}"

        if [[ ! -d "${src}" ]]; then
            die "Source not found: ${src}"
        fi

        if [[ -L "${dst}" ]]; then
            local current
            current="$(readlink -f "${dst}")"
            if [[ "${current}" == "$(readlink -f "${src}")" ]]; then
                info "Already installed: ${dir}"
                skipped=$((skipped + 1))
                continue
            fi
            # Only replace if the existing symlink points into our repo
            if [[ "${current}" == "${SKILLS_SRC}/"* ]]; then
                warn "Stale symlink, replacing: ${dst} -> ${current}"
                rm "${dst}"
            else
                die "${dst} exists but points outside this repo (${current}). Remove it manually if it belongs to us, or rename the skill."
            fi
        elif [[ -e "${dst}" ]]; then
            die "${dst} exists and is not a symlink. Remove it manually first."
        fi

        ln -s "${src}" "${dst}"
        info "Installed: ${dir} -> ${dst}"
        installed=$((installed + 1))
    done

    [ ${cleaned} -gt 0 ] && info "Cleaned up ${cleaned} stale symlink(s)."
    echo ""
    info "Done. Installed: ${installed}, Already up-to-date: ${skipped}"
    echo ""
    info "Available commands in Claude Code:"
    info "  /ncu-profile <kernel.py>   — NCU hardware counter analysis"
    info "  /nsys-profile <kernel.py>  — NSYS timeline analysis"
    info "  /profile <kernel.py>       — Combined NCU + NSYS analysis"
}

do_uninstall() {
    local removed=0

    for dir in "${SKILL_DIRS[@]}"; do
        local dst="${SKILLS_DST}/${dir}"

        if [[ -L "${dst}" ]]; then
            local target
            target="$(readlink "${dst}")"
            if [[ "${target}" != "${SKILLS_SRC}/${dir}" ]]; then
                warn "Symlink points elsewhere, skipping (not ours): ${dst} -> ${target}"
                continue
            fi
            rm "${dst}"
            info "Removed: ${dst}"
            removed=$((removed + 1))
        elif [[ -e "${dst}" ]]; then
            warn "Not a symlink, skipping: ${dst}"
        fi
    done

    info "Removed ${removed} symlink(s)."
}

do_status() {
    echo "Source:  ${SKILLS_SRC}"
    echo "Target:  ${SKILLS_DST}"
    echo ""

    for dir in "${SKILL_DIRS[@]}"; do
        local dst="${SKILLS_DST}/${dir}"
        local src="${SKILLS_SRC}/${dir}"

        if [[ -L "${dst}" ]]; then
            local current
            current="$(readlink -f "${dst}")"
            if [[ "${current}" == "$(readlink -f "${src}")" ]]; then
                echo "  ${dir}: installed"
            else
                echo "  ${dir}: symlink exists but points to ${current}"
            fi
        elif [[ -e "${dst}" ]]; then
            echo "  ${dir}: exists (not a symlink)"
        else
            echo "  ${dir}: not installed"
        fi
    done
}

do_check() {
    local ok=0 warn=0 fail=0

    _check() {
        local label="$1"; shift
        if "$@" &>/dev/null; then
            echo "  [OK]    ${label}"
            ok=$((ok + 1))
        else
            echo "  [FAIL]  ${label}"
            fail=$((fail + 1))
        fi
    }
    _warn() {
        local label="$1"; shift
        if "$@" &>/dev/null; then
            echo "  [OK]    ${label}"
            ok=$((ok + 1))
        else
            echo "  [WARN]  ${label}"
            warn=$((warn + 1))
        fi
    }

    echo "=== GPU ==="
    _check "nvidia-smi available" command -v nvidia-smi
    if command -v nvidia-smi &>/dev/null; then
        echo "         $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1)"
    fi

    echo ""
    echo "=== CUDA Toolkit ==="
    _check "NCU binary"   sh -c 'command -v ncu || ls /usr/local/cuda*/bin/ncu 2>/dev/null || ls /opt/cuda/bin/ncu 2>/dev/null'
    _check "NSYS binary"  sh -c 'command -v nsys || ls /usr/local/cuda*/bin/nsys 2>/dev/null'

    echo ""
    echo "=== Profiling Permissions ==="
    if [ -f /proc/driver/nvidia/params ]; then
        if cat /proc/driver/nvidia/params 2>/dev/null | grep -q "RmProfilingAdminOnly: 0"; then
            echo "  [OK]    RmProfilingAdminOnly=0 — NCU can run without sudo"
        else
            echo "  [WARN]  RmProfilingAdminOnly != 0 — NCU requires sudo"
            echo "          Fix (one-time, needs reboot):"
            echo "            echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf"
            echo "            sudo update-initramfs -u"
            echo "            sudo reboot"
            echo "          Until then, profiling scripts auto-use sudo (may prompt for password)."
        fi
    elif [ -f /etc/modprobe.d/nvidia-profiling.conf ] && grep -q "NVreg_RestrictProfilingToAdminUsers=0" /etc/modprobe.d/nvidia-profiling.conf 2>/dev/null; then
        echo "  [WARN]  nvidia-profiling.conf exists but may need reboot to take effect"
    else
        echo "  [WARN]  Cannot check profiling permissions — NCU may need sudo"
    fi

    echo ""
    echo "=== Python ==="
    _check "Python 3" command -v python3
    _check "torch with CUDA" python3 -c "import torch; assert torch.cuda.is_available()"

    echo ""
    echo "=== Summary ==="
    echo "  OK: ${ok}  WARN: ${warn}  FAIL: ${fail}"
    if [ ${fail} -gt 0 ]; then
        echo "  Fix FAIL items above before profiling."
    fi
    if [ ${warn} -gt 0 ]; then
        echo "  WARN items are non-blocking but may require sudo or reboot."
    fi
}

# Parse flags before command
while [ $# -gt 0 ]; do
    case "$1" in
        -t|--target) TARGET="$2"; shift 2 ;;
        -h|--help)   usage; exit 0 ;;
        *) break ;;
    esac
done

_resolve_target

case "${1:-}" in
    install)    do_install ;;
    uninstall)  do_uninstall ;;
    status)     do_status ;;
    check)      do_check ;;
    "")         do_install ;;
    *)          die "Unknown command: $1 (use -h for help)" ;;
esac
