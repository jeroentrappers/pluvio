#!/bin/bash
# Dead-man's switch for a rented Runcrate GPU: after $1 seconds, DELETE instance
# $2 using the API key in /opt/pluvio/rc.key. Armed via systemd-run on hetz1 so
# the instance is torn down even if the orchestrating session disconnects.
sleep "${1:-14400}"
curl -s -X DELETE -H "Authorization: Bearer $(cat /opt/pluvio/rc.key)" \
  "https://www.runcrate.ai/api/v1/instances/$2" > /tmp/gpu_deadman.log 2>&1
echo "deadman fired for $2 at $(date -u)" >> /tmp/gpu_deadman.log
