#!/usr/bin/env bash
# Keep a local port forwarded to Ollama on a remote box, and RESTART it when it drops.
#
# ⚠ A BARE `ssh -N -L` IS NOT ENOUGH, and the failure is silent downstream. One died mid-run and
# every request after it was refused instantly; the glosser treated each refusal as a hard sentence
# and fell back to the dictionary, writing four complete-looking `-llm.json` files that were 100 %
# Wiktionary. The glosser now aborts on an unreachable server -- this keeps it reachable.
set -u
REMOTE=${REMOTE:-skmm}
LPORT=${LPORT:-11435}
RPORT=${RPORT:-11434}
while true; do
  ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=20 -o ServerAliveCountMax=3 \
      -o ConnectTimeout=10 -L "$LPORT:localhost:$RPORT" "$REMOTE"
  echo "$(date -u +%FT%TZ) tunnel to $REMOTE dropped (exit $?), reconnecting in 5s" >&2
  sleep 5
done
