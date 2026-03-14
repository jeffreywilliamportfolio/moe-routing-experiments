#!/usr/bin/env bash
# setup_instance.sh — Bootstrap 4x H200 Vast.ai instance for Ling-1T 5-cond dog capture.
#
# Run this script ON THE INSTANCE after SSH in:
#   bash /workspace/experiment-ling1t-dog/setup_instance.sh
#
# Phases:
#   A — start model download in background screen session
#   B — install deps + clone llama.cpp + build capture binary (parallel with download)
#   C — verify tensor name (requires model download complete)
#   D — generate TSV, token_verify, then main run
#   E — print download command, then kill instance from local machine

set -euo pipefail
WORKDIR=/workspace/experiment-ling1t-dog
LOG="$WORKDIR/setup.log"
mkdir -p "$WORKDIR"
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo " Ling-1T 5-Cond Dog Instance Bootstrap — $(date)"
echo "============================================================"

MODEL_DIR=/workspace/models/Ling-1T-Q3_K_S
LLAMA_DIR=/workspace/src/llama.cpp-bailingmoe2
BUILD_DIR="$LLAMA_DIR/build-cuda"
BINARY=/workspace/consciousness-experiment/capture_activations_ling1t

# ----------------------------------------------------------------
# PHASE A — kick off model download in background screen session
# ----------------------------------------------------------------
echo ""
echo "=== PHASE A: Start model download ==="
pip install -q hf_transfer huggingface_hub
mkdir -p "$MODEL_DIR"

screen -dmS download bash -c "
  set -euo pipefail
  HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
    unsloth/Ling-1T-GGUF \
    --include 'Ling-1T-UD-Q3_K_S*' \
    --local-dir '$MODEL_DIR' \
    --local-dir-use-symlinks False \
  2>&1 | tee '$WORKDIR/download.log'
  echo 'DOWNLOAD COMPLETE' >> '$WORKDIR/download.log'
"
echo "Download started in screen session 'download'."
echo "Monitor: screen -r download   or   tail -f $WORKDIR/download.log"

# ----------------------------------------------------------------
# PHASE B — system deps + clone + build (parallel with download)
# ----------------------------------------------------------------
echo ""
echo "=== PHASE B: System deps ==="
apt-get update -qq && apt-get install -y -qq \
  git build-essential cmake ninja-build pkg-config libcurl4-openssl-dev
pip install -q numpy scipy

echo ""
echo "=== PHASE B: Clone llama.cpp master ==="
if [ ! -d "$LLAMA_DIR/.git" ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
else
  echo "  Already cloned, skipping."
fi
git -C "$LLAMA_DIR" rev-parse HEAD > "$WORKDIR/build_commit.txt"
echo "  Commit: $(cat $WORKDIR/build_commit.txt)"

echo ""
echo "=== PHASE B: Inject capture_activations into build system ==="
mkdir -p "$LLAMA_DIR/examples/capture_activations"
cp "$WORKDIR/capture_activations.cpp" "$LLAMA_DIR/examples/capture_activations/"

cat > "$LLAMA_DIR/examples/capture_activations/CMakeLists.txt" << 'CMEOF'
set(TARGET llama-capture-activations)
add_executable(${TARGET} capture_activations.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
CMEOF

if ! grep -q 'add_subdirectory(capture_activations)' "$LLAMA_DIR/examples/CMakeLists.txt"; then
  echo 'add_subdirectory(capture_activations)' >> "$LLAMA_DIR/examples/CMakeLists.txt"
  echo "  Added capture_activations to examples/CMakeLists.txt"
else
  echo "  Already present in examples/CMakeLists.txt"
fi

echo ""
echo "=== PHASE B: Build ==="
cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  2>&1 | tee "$WORKDIR/cmake.log"

cmake --build "$BUILD_DIR" --target llama-capture-activations -j$(nproc) \
  2>&1 | tee "$WORKDIR/build.log"

mkdir -p /workspace/consciousness-experiment
cp "$BUILD_DIR/bin/llama-capture-activations" "$BINARY"
md5sum "$BINARY" > "$WORKDIR/binary_md5.txt"
echo "  Binary: $BINARY"
echo "  MD5:    $(cat $WORKDIR/binary_md5.txt)"

# ----------------------------------------------------------------
# PHASE C — tensor name verification (requires model present)
# ----------------------------------------------------------------
echo ""
echo "=== PHASE C: Waiting for model download to complete ==="
echo "  (tail -f $WORKDIR/download.log to monitor)"
echo ""

while ! grep -q "DOWNLOAD COMPLETE" "$WORKDIR/download.log" 2>/dev/null; do
  printf '.'
  sleep 30
done
echo ""
echo "  Download complete."

MODEL=$(find "$MODEL_DIR" -name "*.gguf" 2>/dev/null | sort | head -1)
if [ -z "$MODEL" ]; then
  echo "ERROR: No .gguf files found in $MODEL_DIR"
  exit 1
fi
echo "  Model shard: $MODEL"

echo ""
echo "=== PHASE C: Tensor name verification ==="
LD_LIBRARY_PATH="$BUILD_DIR/bin:${LD_LIBRARY_PATH:-}" \
  "$BINARY" \
  -m "$MODEL" -p "Hello" -o /tmp/ling1t_test \
  -n 0 -ngl 999 -c 512 -t 16 \
  --list-tensors 2>&1 \
  | grep -i "moe\|gate\|router\|logit" | head -30 \
  | tee "$WORKDIR/tensor_names.txt"

echo ""
echo "--- tensor_names.txt saved to $WORKDIR/tensor_names.txt ---"
echo "Verify: lines should show ffn_moe_logits-N with dims=[N_TOKENS, 256]."

# ----------------------------------------------------------------
# PHASE D — generate TSV + token_verify + main run
# ----------------------------------------------------------------
echo ""
echo "=== PHASE D: Generate TSV ==="
cd "$WORKDIR"
python3 generate_tsv.py

echo ""
echo "=== PHASE D: Main run (150 prompts) ==="
NGL=999 \
CAPTURE_BINARY="$BINARY" \
MODEL_PATH="$MODEL" \
LLAMA_BUILD_BIN="$BUILD_DIR/bin" \
  python3 run_experiment.py 2>&1 | tee "$WORKDIR/experiment.log"

# ----------------------------------------------------------------
# PHASE E — print download command
# ----------------------------------------------------------------
echo ""
echo "============================================================"
echo " PHASE E: Run complete. Download from local machine:"
echo "============================================================"
echo ""
echo "  scp -P PORT -i ~/.ssh/vast_16 -r \\"
echo "    root@HOST:/workspace/experiment-ling1t-dog/output/ \\"
echo "    experiments/ling1t-5cond-dog-1/"
echo ""
echo "  scp -P PORT -i ~/.ssh/vast_16 \\"
echo "    root@HOST:/workspace/experiment-ling1t-dog/{experiment.log,build_commit.txt,binary_md5.txt,tensor_names.txt,setup.log} \\"
echo "    experiments/ling1t-5cond-dog-1/"
echo ""
echo "  Then kill the instance and run:"
echo "    python3 experiments/ling1t-5cond-dog-1/analyze_local.py \\"
echo "      --output-dir experiments/ling1t-5cond-dog-1/output/ \\"
echo "      --prompt-suite experiments/ling1t-5cond-dog-1/prompt_suite.json"
echo ""
echo "============================================================"
echo " Setup complete — $(date)"
echo "============================================================"
