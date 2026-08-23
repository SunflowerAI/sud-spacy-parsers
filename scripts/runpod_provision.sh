#!/usr/bin/env bash
# Provision a RunPod GPU pod to build and train the en_gum arm. IDEMPOTENT -- safe to re-run, and
# meant to be: if a pod dies mid-run, create another and run this again.
#
# WHAT THE GPU IS ACTUALLY FOR, stated plainly because it decides how you read the timings.
# These are small CNN pipelines with no transformer, and spaCy's GPU speedup on them is typically
# 1.5-2x and can be NEGATIVE (host<->device transfer dominates at small batch). `sud_shared` is
# worse than that: sud.HeadDepsTagger.v1's forward pass runs a per-token Python loop doing
# `X[idx].mean(axis=0)`, i.e. thousands of tiny kernel launches per document. The unambiguous GPU
# win here is OLLAMA -- qwen3:32b at usable speed is simply not available on an M-series Mac.
# So: probe, then use whichever device wins per arm, and report the numbers rather than assuming.
#
# Usage (on the pod):  bash /workspace/runpod_provision.sh
set -uo pipefail
ROOT=${ROOT:-/workspace/SUD-spaCy}
REPO=${REPO:-https://github.com/SunflowerAI/sud-spacy-parsers.git}
BRANCH=${BRANCH:-en-ewt-gum}
LOGS=${LOGS:-/workspace/logs}
export DEBIAN_FRONTEND=noninteractive
mkdir -p "$LOGS"
die () { echo "FATAL: $*" >&2; exit 1; }

echo "=== 0. GPU preflight (before anything is installed or billed against) =========="
command -v nvidia-smi >/dev/null || die "no nvidia-smi -- this is not a GPU pod"
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
# spaCy 3.8 pins cupy<13 for its cuda12x extra, and cupy 12.x ships no Blackwell (sm_100/sm_120)
# kernels. On such a card everything installs, the import succeeds, and the FIRST KERNEL LAUNCH
# fails -- potentially hours in. Refuse here instead. Ampere (8.x) / Ada (8.9) / Hopper (9.0) are
# all fine; A6000 is 8.6.
MAJOR=${CC%%.*}
if [ "${MAJOR:-0}" -ge 10 ]; then
  die "compute capability $CC is Blackwell or newer. spaCy 3.8 pins cupy<13, which has no kernels
     for it -- the failure would surface at the first kernel launch, not at install. Terminate this
     pod and take an A6000 / A5000 / L40S / A100 (compute_cap 8.x or 9.0)."
fi
[ "$CC" = "8.6" ] || echo "  NOTE: expected 8.6 (RTX A6000); got $CC. Supported, but the cost
  estimate assumed A6000 pricing."

echo "=== 1. base packages ==========================================================="
apt-get update -qq && apt-get install -y -qq tmux rsync curl git ca-certificates bc \
  || die "apt failed"

echo "=== 2. repo ===================================================================="
# Two ways in, and the pod must not care which. CLONE if nothing is here (public repo, so no
# credential ever lands on the pod -- which is what makes terminating it a zero-risk operation).
# But an RSYNC of the working tree is equally valid and is the route when the branch has not been
# pushed; in that case there is nothing to fetch and trying would fail the run.
if [ ! -d "$ROOT/.git" ]; then
  git clone "$REPO" "$ROOT" || die "clone failed (rsync the working tree here instead)"
  cd "$ROOT" && { git checkout "$BRANCH" || die "no branch $BRANCH on the remote"; }
else
  cd "$ROOT" || die "no $ROOT"
  if git fetch origin "$BRANCH" 2>/dev/null; then
    git checkout "$BRANCH" && git pull --ff-only || die "checkout failed"
  else
    echo "  no remote branch $BRANCH -- using the working tree as delivered (rsync route)"
  fi
fi
echo "  at $(git log --oneline -1 2>/dev/null || echo '(no git metadata)')"

echo "=== 3. python 3.12 + spaCy with CUDA ==========================================="
# The RunPod images ship 3.11; uv fetches a standalone CPython 3.12 with no apt/PPA involved.
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh || die "uv install failed"
  export PATH="$HOME/.local/bin:$PATH"
fi
[ -x "$ROOT/.venv/bin/python" ] || uv venv --python 3.12 "$ROOT/.venv" || die "venv failed"
PY="$ROOT/.venv/bin/python"
"$ROOT/.venv/bin/pip" install -q -U pip
# requirements.txt's `thinc-apple-ops; platform_machine == "arm64"` correctly no-ops on Linux x86_64.
"$ROOT/.venv/bin/pip" install -q -r "$ROOT/requirements.txt" || die "requirements failed"
"$ROOT/.venv/bin/pip" install -q "spacy[cuda12x]" || die "spacy[cuda12x] failed"

echo "=== 4. GPU smoke tests (cheap, and each one has cost someone a long run) ========"
$PY - <<'PYEOF' || die "GPU smoke test failed"
import cupy, spacy, thinc
print(f"  spacy {spacy.__version__}  thinc {thinc.__version__}  cupy {cupy.__version__} "
      f"(cuda runtime {cupy.cuda.runtime.runtimeGetVersion()})")
assert int(cupy.__version__.split(".")[0]) < 13, "cupy >= 13 -- outside spaCy 3.8's pin"
print("  require_gpu:", spacy.require_gpu())
from thinc.api import get_current_ops
ops = get_current_ops()
print("  ops:", type(ops).__name__)
# The one that would otherwise fail in BACKPROP, minutes into the longest run of the chain.
xp = ops.xp
t = ops.alloc2f(4, 3)
i = xp.asarray([0, 0, 2], dtype="i")
v = xp.ones((3, 3), dtype="f")
ops.scatter_add(t, i, v)
got = t.get().tolist() if hasattr(t, "get") else t.tolist()
assert got[0] == [2.0, 2.0, 2.0] and got[2] == [1.0, 1.0, 1.0], got
print("  ops.scatter_add accumulates correctly on this backend:", got[0], got[2])
PYEOF

echo "=== 5. treebanks ==============================================================="
mkdir -p "$ROOT/assets"
for tb in EWT GUM; do
  f="$ROOT/assets/SUD_English-$tb.tgz"
  [ -s "$f" ] || curl -fsSL -o "$f" "https://grew.fr/download/SUD_2.18/SUD_English-$tb.tgz" \
    || die "download of SUD_English-$tb.tgz failed -- rsync the local copy instead"
  [ -d "$ROOT/assets/SUD_English-$tb" ] || tar xzf "$f" -C "$ROOT/assets"
done
sha256sum "$ROOT"/assets/SUD_English-*.tgz
# Expected, from the copies verified locally:
#   a3292935f56897f0d4066cb56028cb64b1e68298080682b3a8d576a96d13f07b  SUD_English-EWT.tgz
#   7e8ef52d814f72a6a7261ea7b786cebd342b6602aa0ea36c2ac5627298d3fdbb  SUD_English-GUM.tgz

echo "=== 6. ollama =================================================================="
command -v ollama >/dev/null || curl -fsSL https://ollama.com/install.sh | sh || die "ollama failed"
mkdir -p /workspace/ollama
# NOT started here -- see the serve line below, which must set the env that keeps the prompt-prefix
# KV cache alive. Started under tmux so it survives an SSH drop.
cat <<'EOS'
  ollama is installed but NOT running. Start it with:

    tmux new -d -s ollama '
      OLLAMA_MODELS=/workspace/ollama \
      OLLAMA_HOST=127.0.0.1:11434 \
      OLLAMA_KEEP_ALIVE=-1 \
      OLLAMA_NUM_PARALLEL=1 \
      OLLAMA_MAX_LOADED_MODELS=1 \
      ollama serve 2>&1 | tee -a /workspace/logs/ollama.log'

  Every one of those settings is load-bearing:
    127.0.0.1  RunPod pod IPs are PUBLIC and ollama has no auth. Never expose 11434.
    KEEP_ALIVE=-1  the default 5-minute unload tears down the slot AND its cached prompt prefix;
                   the relabel prompts are a long static prefix + a short suffix precisely so that
                   cache is reused, so an unload during any pause costs a full re-prefill.
    NUM_PARALLEL=1 relabel_ext.py is a single-threaded blocking loop. The server cannot be parallel
                   if the client is serial; raising this only splits the KV cache across idle slots.
    MAX_LOADED=1   qwen3:8b and :32b must not be co-resident, or the comparison timings are noise.
EOS

echo "=== 7. done ===================================================================="
echo "PROVISION OK $(date -Is)  $(git -C "$ROOT" log --oneline -1)" | tee "$LOGS/PROVISION.ok"
cat <<'EOS'

  Still to come from the Mac (gitignored, so `git clone` cannot bring it):
    rsync gold/gold_udep.jsonl     -- the 400-item comp/mod benchmark; without it the 8b-vs-32b
                                 comparison has no gold to score against.

  Then:
    bash scripts/build_en_ewt_gum.sh all       # data; must report 0 model calls
    MAX_STEPS=200 GPU_ID=0 OUT_DIR=/tmp/probe_gpu LOG=/tmp/probe_gpu.log \
      bash scripts/build_en_ewt_gum.sh base    # and again without GPU_ID; keep the faster
EOS
