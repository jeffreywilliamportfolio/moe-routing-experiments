# Ling-1T (BailingMoeV2) — Capture Binary Prep

## Architecture

| Property | Value |
|----------|-------|
| Model | Ling-1T (inclusionAI/Ling-1T) |
| Architecture | BailingMoeV2 (`bailingmoe2`) |
| Total params | 1T |
| Active params/token | ~50B |
| Total layers | 80 |
| Dense layers | First 4 (layers 0–3) |
| MoE layers | Layers 4–79 (~76; verify with --list-tensors) |
| Routed experts | 256 |
| Shared experts | 1 (always active, not in routing tensor) |
| Active experts/token | 8 (top-k) |
| Gating function | **Sigmoid** (NOT softmax) |
| Context | 32K native, 128K with YaRN |
| Vocab size | 157,184 |
| add_bos_token | false |

Target GGUF: `unsloth/Ling-1T-GGUF` Q3_K_S (~431GB) — requires 4× H200.

## Chat Template

Single-turn inference (no system message body, thinking disabled):

```
<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>{prompt}<|role_end|><role>ASSISTANT</role>
```

Key special tokens:
- `<|role_end|>` — EOS / turn separator
- `<role>HUMAN</role>`, `<role>ASSISTANT</role>`, `<role>SYSTEM</role>` — role markers
- No BOS token (add_bos_token: false in tokenizer config)

**Verify this template on the instance before running.** Check with:
```bash
python3 -c "
from llama_cpp import Llama
m = Llama(model_path='<gguf>', verbose=False)
print(m.metadata.get('tokenizer.chat_template', 'NOT FOUND'))
"
```
Or inspect the GGUF metadata: `gguf-dump <gguf> | grep chat_template`.

## LLM Support Status

BailingMoeV2 is **not in llama.cpp b8123**. Support is in PR #16063 by user CISC:
- PR: https://github.com/ggml-org/llama.cpp/pull/16063
- Branch: `cisc/bailingmoe2`
- ubergarm has verified inference from this branch: https://huggingface.co/ubergarm/Ling-1T-GGUF

## Tensor Name Analysis

**Finding: No changes needed to `is_router_tensor()`.**

BailingMoeV2 routes through the shared `build_moe_ffn()` function in `llama-graph.cpp`, which calls:
```cpp
cb(logits, "ffn_moe_logits", il);
```

The `cb()` lambda formats this as `ffn_moe_logits-{il}` (e.g., `ffn_moe_logits-4` through `ffn_moe_logits-79`), identical to DeepSeek/Qwen/GLM naming. Our existing filter:
```cpp
static bool is_router_tensor(const char * name) {
    return strstr(name, "ffn_moe_logits") != nullptr;
}
```
will match these tensors unchanged.

**Verify on first run** by passing `--list-tensors` and confirming:
1. Tensors appear as `ffn_moe_logits-4` through `ffn_moe_logits-79` (or similar range)
2. Count matches expected MoE layer count
3. Shape is `[n_tokens, 256]` (not `[n_tokens, 257]` — shared expert is NOT in routing tensor)

The BailingMoeV2-only tensor `ffn_moe_weights_sum_biased` does not match `is_router_tensor()` and will not be captured. This is correct behavior.

## capture_activations.cpp Changes Required

**C++ logic changes: NONE.**

The source at `results/2026-03-06-ds31-v22-32q-1/scripts/capture_activations.cpp` (md5: `59a5f9952194536747229e033fc93ca5`) can be used as-is. All existing architecture-specific behaviors are handled by the model's compute graph — the binary only intercepts named tensors and is fully architecture-agnostic.

**Potential compile-time issues to fix if they arise:**

1. `#include "llama-cpp.h"` — this header may not exist in cisc/bailingmoe2. Remove it if the build fails; it was a b8123 convenience include and nothing in our source actually uses symbols from it exclusively.

2. `llama_vocab_n_tokens(vocab)` — in some newer llama.cpp versions this is `llama_n_vocab(model)`. Fix at compile time.

3. Any `common_params` field name changes — check `common/common.h` on the instance and fix.

None of these changes alter the capture logic.

## Build Instructions (on instance)

```bash
# 1. Clone cisc/bailingmoe2 branch
git clone --depth=1 -b cisc/bailingmoe2 \
  https://github.com/ggml-org/llama.cpp \
  /workspace/llama.cpp-bailingmoe2

# 2. Record the commit hash for paper provenance
cd /workspace/llama.cpp-bailingmoe2
git rev-parse HEAD > /workspace/experiment-ling1t/build_commit.txt
cat /workspace/experiment-ling1t/build_commit.txt

# 3. Copy capture binary source into the examples directory
mkdir -p examples/capture_activations
cp /workspace/experiment-ling1t/capture_activations.cpp examples/capture_activations/

# 4. Create CMakeLists for the example target
cat > examples/capture_activations/CMakeLists.txt << 'EOF'
llama_build_and_install_target(llama-capture-activations capture_activations.cpp)
EOF

# (If that macro doesn't exist in this llama.cpp version, use:)
# add_executable(llama-capture-activations capture_activations.cpp)
# target_link_libraries(llama-capture-activations PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})

# 5. Add the subdirectory to examples/CMakeLists.txt
echo 'add_subdirectory(capture_activations)' >> examples/CMakeLists.txt

# 6. Configure and build
export LD_LIBRARY_PATH=/workspace/llama.cpp-bailingmoe2/build/bin:$LD_LIBRARY_PATH
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-capture-activations -j$(nproc)

# 7. Deploy binary
cp build/bin/llama-capture-activations /workspace/consciousness-experiment/capture_activations_ling1t

# 8. Record md5
md5sum /workspace/consciousness-experiment/capture_activations_ling1t \
  > /workspace/experiment-ling1t/binary_md5.txt
cat /workspace/experiment-ling1t/binary_md5.txt
```

**Do NOT overwrite `/workspace/consciousness-experiment/capture_activations`** — that binary is the production binary used across all prior experiments. The Ling-1T binary is a separate file.

## Verification Run

Before the full 60-prompt experiment, run a single-prompt sanity check:

```bash
# Create a 1-line TSV for testing
echo -e "TEST_01\t<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>What is 2+2?<|role_end|><role>ASSISTANT</role>" \
  > /workspace/experiment-ling1t/test_single.tsv

# Sanity run with --list-tensors first
LD_LIBRARY_PATH=/workspace/llama.cpp-bailingmoe2/build/bin \
/workspace/consciousness-experiment/capture_activations_ling1t \
  -m /workspace/models/Ling-1T-Q3_K_S/Ling-1T-Q3_K_S-00001-of-00NNN.gguf \
  --prompt-file /workspace/experiment-ling1t/test_single.tsv \
  -o /workspace/experiment-ling1t/test_output \
  -n 0 -ngl 999 -c 2048 -t 16 \
  --list-tensors

# Then capture run
LD_LIBRARY_PATH=/workspace/llama.cpp-bailingmoe2/build/bin \
/workspace/consciousness-experiment/capture_activations_ling1t \
  -m /workspace/models/Ling-1T-Q3_K_S/Ling-1T-Q3_K_S-00001-of-00NNN.gguf \
  --prompt-file /workspace/experiment-ling1t/test_single.tsv \
  -o /workspace/experiment-ling1t/test_output \
  -n 0 -ngl 999 -c 2048 -t 16 \
  --routing-only --no-stream
```

Check:
- `ls test_output/TEST_01/router/` — should show `ffn_moe_logits-*.npy` files
- Count: `ls test_output/TEST_01/router/ | wc -l` — should be ~76
- Shape: `python3 -c "import numpy as np; import glob; f=sorted(glob.glob('test_output/TEST_01/router/*.npy'))[0]; print(np.load(f).shape)"` — should be `(N_TOKENS, 256)`

## Issues Flagged

### Issue 1: Sigmoid routing — Python analysis must change (out of scope here)

Every prior model uses softmax routing. The Python analysis scripts apply `softmax()` to the captured logits before computing Shannon entropy. For Ling-1T, the model uses sigmoid scoring:

```
score_i = sigmoid(logit_i)  # NOT softmax
```

The captured logits are pre-sigmoid raw gate scores. The Python `compute_metrics()` function must be modified for Ling-1T to:
```python
# For sigmoid-routed models:
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def compute_probs_sigmoid(logits):
    scores = sigmoid(logits)
    return scores / scores.sum(axis=-1, keepdims=True)  # normalize to sum=1
```

This changes the absolute entropy values (sigmoid-then-normalize gives different distribution than softmax) but the A>B directional test should be unaffected. **Do not apply softmax to Ling-1T logits.** Flag the `run_experiment.py` with a TODO at the softmax call.

### Issue 2: MTP layers

Ling-1T has NextN/MTP (multi-token prediction) layers appended to the main stack. In the cisc/bailingmoe2 build, these are loaded with `TENSOR_SKIP` flags and excluded from inference compute graphs. They should NOT produce `ffn_moe_logits` tensors during the eval callback. Verify: the `--list-tensors` run should show exactly ~76 unique `ffn_moe_logits-N` tensor names (not 80+).

If MTP layers DO produce additional tensors, the layer count will be higher than expected. The existing row-count exclusion logic in `compute_metrics()` (median ×0.5 threshold) will catch them only if they have truncated rows — not guaranteed. In that case, add the MTP layer indices to `EXCLUDED_LAYERS` explicitly after inspecting the captured shapes.

### Issue 3: Token count verification

The prompt suite uses `" The routing process continues through subsequent layers without interruption."` as padding. For Ling-1T's tokenizer (157K vocab, Qwen-derived BPE), this sentence likely tokenizes to a different count than for DeepSeek. Run `token_verify.py` — adapted for the Ling-1T template and tokenizer — before the main experiment run.

The PAD_WORD for single-token adjustments needs verification. DeepSeek uses the full pad sentence; GPT-OSS used `" Also"`. Ling-1T needs its own pad word verified against its tokenizer.

### Issue 4: Known-bad llama.cpp header (low risk)

`#include "llama-cpp.h"` in capture_activations.cpp was a b8123-specific header. In cisc/bailingmoe2 it may be absent. If the build fails with "file not found", remove that include line — no symbols from it are needed.

## Binary Provenance Record

For the paper, document:
- Base: llama.cpp branch `cisc/bailingmoe2`, commit `<from build_commit.txt>`
- Source: `capture_activations.cpp` (unchanged from md5 `59a5f9952194536747229e033fc93ca5`)
- New binary md5: `<from binary_md5.txt>`
- Differentiated from production binary by filename: `capture_activations_ling1t`
