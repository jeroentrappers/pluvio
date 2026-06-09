#!/usr/bin/env bash
# Install (or re-install) the pluvio forward-pull + rotate systemd-user units.
#
# Usage:
#   tools/install_timers.sh             # install + enable + start
#   tools/install_timers.sh disable     # stop + disable, but leave files in place
#   tools/install_timers.sh uninstall   # remove from ~/.config/systemd/user
#
# Idempotent. Reads the templates from tools/systemd/ and symlinks them into
# ~/.config/systemd/user/. After install, check status with:
#
#   systemctl --user list-timers --all | grep pluvio
#   journalctl --user -u pluvio-pull@knmi-radar -n 30 --no-pager
#
# Timers stay armed across reboots iff your user has `loginctl enable-linger`.
# Without linger, timers fire only while you're logged in.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO_ROOT/tools/systemd"
DST="$HOME/.config/systemd/user"

UNITS=(
    "pluvio-pull@.service"
    "pluvio-pull-knmi-radar.timer"
    "pluvio-pull-kmi-aws.timer"
    "pluvio-pull-meteosat.timer"
    "pluvio-pull-alaro.timer"
    "pluvio-rotate.service"
    "pluvio-rotate.timer"
)

TIMERS=(
    "pluvio-pull-knmi-radar.timer"
    "pluvio-pull-kmi-aws.timer"
    "pluvio-pull-meteosat.timer"
    "pluvio-pull-alaro.timer"
    "pluvio-rotate.timer"
)

case "${1:-install}" in
    install)
        mkdir -p "$DST"
        for u in "${UNITS[@]}"; do
            ln -sf "$SRC/$u" "$DST/$u"
            echo "linked $u"
        done
        systemctl --user daemon-reload
        for t in "${TIMERS[@]}"; do
            systemctl --user enable --now "$t"
        done
        echo
        echo "Enable persistence across logouts: loginctl enable-linger $USER"
        echo "Status:                            systemctl --user list-timers --all | grep pluvio"
        ;;
    disable)
        for t in "${TIMERS[@]}"; do
            systemctl --user disable --now "$t" || true
        done
        systemctl --user daemon-reload
        ;;
    uninstall)
        for t in "${TIMERS[@]}"; do
            systemctl --user disable --now "$t" || true
        done
        for u in "${UNITS[@]}"; do
            rm -f "$DST/$u"
        done
        systemctl --user daemon-reload
        ;;
    *)
        echo "usage: $0 [install|disable|uninstall]" >&2
        exit 2
        ;;
esac
