#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "sampling.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <set>

namespace fs = std::filesystem;

// ================================================================
// NumPy .npy writer (v1.0, float32, C-contiguous)
// ================================================================
static bool write_npy(const std::string & path, const float * data,
                      const std::vector<int64_t> & shape) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    std::string shape_str = "(";
    for (size_t i = 0; i < shape.size(); i++) {
        shape_str += std::to_string(shape[i]);
        if (i + 1 < shape.size() || shape.size() == 1) shape_str += ",";
    }
    shape_str += ")";

    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape_str + ", }";
    int total = 10 + (int)header.size() + 1;
    int pad = 64 - (total % 64);
    if (pad == 64) pad = 0;
    header += std::string(pad, ' ');
    header += '\n';

    uint16_t header_len = (uint16_t)header.size();
    f.write("\x93NUMPY", 6);
    uint8_t ver[2] = {1, 0};
    f.write((char *)ver, 2);
    f.write((char *)&header_len, 2);
    f.write(header.data(), header_len);

    size_t n_elements = 1;
    for (auto s : shape) n_elements *= s;
    f.write((const char *)data, n_elements * sizeof(float));
    return f.good();
}

static std::string json_escape(const std::string & in) {
    std::string out;
    out.reserve(in.size() + 8);
    for (unsigned char c : in) {
        switch (c) {
            case '\"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back((char)c);
                }
                break;
        }
    }
    return out;
}

// ================================================================
// Tensor accumulator — buffers data across decode calls
// ================================================================
struct tensor_record {
    std::string name;
    std::vector<float> data;   // flat [n_tokens_accum, dim]
    int64_t dim;               // feature dimension (ne[0])
    int64_t n_tokens;          // accumulated token count
    bool    is_router;         // true → save to router/, false → layers/
};

// ================================================================
// Capture state — shared between main loop and callback
// ================================================================
struct capture_state {
    std::map<std::string, tensor_record> tensors;
    std::set<std::string> discovered_tensor_names;
    std::set<std::string> logged_shape_mismatch;
    std::string output_dir;
    std::string current_prompt_id;
    bool    prompt_has_error   = false;
    bool    list_tensors_mode = false;
    bool    routing_only      = false;
    bool    verbose           = false;
    int     tensors_seen      = 0;
    int     tensors_captured  = 0;
    std::vector<uint8_t> xfer_buf;   // reusable GPU→CPU transfer buffer
};

// ================================================================
// Tensor name classification
// ================================================================
static bool is_layer_tensor(const char * name) {
    return strstr(name, "l_out")          ||
           strstr(name, "attn_out")       ||
           strstr(name, "ffn_out")        ||
           strstr(name, "result_output")  ||
           strstr(name, "output_norm");
}

static bool is_router_tensor(const char * name) {
    // Only match the actual MoE router logits (ffn_moe_logits-N)
    // These are [n_tokens, 256] tensors — the routing decisions
    return strstr(name, "ffn_moe_logits") != nullptr;
}

static bool should_capture(const char * name) {
    return is_layer_tensor(name) || is_router_tensor(name);
}

// ================================================================
// Float extraction (handles F32 / F16 / BF16)
// ================================================================
static float get_f32(const uint8_t * data, ggml_type type, const size_t * nb,
                     int64_t i0, int64_t i1) {
    size_t off = (size_t)(i1 * nb[1] + i0 * nb[0]);
    switch (type) {
        case GGML_TYPE_F32:  return *(const float         *)&data[off];
        case GGML_TYPE_F16:  return ggml_fp16_to_fp32(*(const ggml_fp16_t *)&data[off]);
        case GGML_TYPE_BF16: return ggml_bf16_to_fp32(*(const ggml_bf16_t *)&data[off]);
        default:             return 0.0f;
    }
}

// ================================================================
// Eval callback — fires for every tensor in the compute graph
// ================================================================
static bool capture_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (capture_state *)user_data;

    // ---- ask phase: tell scheduler which tensors we want ----
    if (ask) {
        if (st->list_tensors_mode) return true;          // grab everything in discovery mode
        if (st->routing_only) return is_router_tensor(t->name);
        return should_capture(t->name);
    }

    st->tensors_seen++;

    // ---- discovery mode: just print tensor info ----
    if (st->list_tensors_mode) {
        st->discovered_tensor_names.insert(t->name);
        int nd = ggml_n_dims(t);
        LOG("[TENSOR] %-50s  type=%-6s  dims=[", t->name, ggml_type_name(t->type));
        for (int d = 0; d < nd; d++) LOG("%lld%s", (long long)t->ne[d], d < nd - 1 ? ", " : "");
        LOG("]  %s%s\n",
            is_layer_tensor(t->name)  ? "LAYER"  : "",
            is_router_tensor(t->name) ? "ROUTER" : "");
        return true;
    }

    // ---- skip quantized tensors (need dequant which is expensive) ----
    if (ggml_is_quantized(t->type)) return true;

    // ---- transfer from GPU if needed ----
    const bool on_host = ggml_backend_buffer_is_host(t->buffer);
    if (!on_host) {
        size_t nb = ggml_nbytes(t);
        st->xfer_buf.resize(nb);
        ggml_backend_tensor_get(t, st->xfer_buf.data(), 0, nb);
    }
    uint8_t * src = on_host ? (uint8_t *)t->data : st->xfer_buf.data();

    // ---- extract dimensions ----
    int64_t dim      = t->ne[0];                                   // feature dim
    int64_t n_tokens = (ggml_n_dims(t) > 1) ? t->ne[1] : 1;      // tokens in this batch

    // ---- convert to f32 in row-major [n_tokens, dim] order ----
    size_t chunk = (size_t)(n_tokens * dim);
    std::vector<float> f32(chunk);
    for (int64_t tok = 0; tok < n_tokens; tok++) {
        for (int64_t d = 0; d < dim; d++) {
            f32[tok * dim + d] = get_f32(src, t->type, t->nb, d, tok);
        }
    }

    // ---- accumulate into tensor record ----
    std::string name(t->name);
    auto it = st->tensors.find(name);
    if (it != st->tensors.end()) {
        // subsequent decode (generation) — append tokens
        auto & rec = it->second;
        if (rec.dim != dim) {
            if (st->logged_shape_mismatch.insert(name).second) {
                LOG_ERR("  tensor dim mismatch for %s: expected %lld got %lld\n",
                        name.c_str(), (long long)rec.dim, (long long)dim);
            }
            st->prompt_has_error = true;
            return true;
        }
        rec.data.insert(rec.data.end(), f32.begin(), f32.end());
        rec.n_tokens += n_tokens;
    } else {
        tensor_record rec;
        rec.name      = name;
        rec.data      = std::move(f32);
        rec.dim       = dim;
        rec.n_tokens  = n_tokens;
        rec.is_router = is_router_tensor(t->name);
        st->tensors[name] = std::move(rec);
    }

    st->tensors_captured++;
    return true;
}

// ================================================================
// Save all captured tensors for one prompt
// ================================================================
struct generated_token_info {
    int32_t     token_id;
    std::string piece;
};

static void save_prompt(capture_state & st, const std::string & prompt_text,
                        const std::string & generated_text,
                        const std::vector<generated_token_info> & generated_tokens,
                        int n_tok_prompt, int n_tok_gen, double ms) {
    fs::path base = fs::path(st.output_dir);
    int n_layer = 0, n_router = 0;

    for (auto & [name, rec] : st.tensors) {
        fs::path dir = base / (rec.is_router ? "router" : "layers");
        fs::create_directories(dir);

        // sanitize filename
        std::string fname = name;
        for (char & c : fname) {
            if (c == '/' || c == ' ' || c == '.') c = '_';
        }

        std::vector<int64_t> shape = {rec.n_tokens, rec.dim};
        if (write_npy((dir / (fname + ".npy")).string(), rec.data.data(), shape)) {
            if (rec.is_router) n_router++; else n_layer++;
        }
    }

    // metadata
    {
        std::ofstream f((base / "metadata.txt").string());
        std::string esc = prompt_text;
        std::replace(esc.begin(), esc.end(), '\n', ' ');
        f << "prompt_id="          << st.current_prompt_id << "\n"
          << "prompt="             << esc                   << "\n"
          << "n_tokens_prompt="    << n_tok_prompt          << "\n"
          << "n_tokens_generated=" << n_tok_gen             << "\n"
          << "n_layer_tensors="    << n_layer               << "\n"
          << "n_router_tensors="   << n_router              << "\n"
          << "elapsed_ms="         << (int)ms               << "\n";
    }

    // generated text — always saved to file regardless of --no-stream
    if (!generated_text.empty()) {
        std::ofstream f((base / "generated_text.txt").string());
        f << generated_text;
    }

    {
        std::ofstream f((base / "generated_tokens.json").string());
        f << "[\n";
        for (size_t i = 0; i < generated_tokens.size(); i++) {
            const auto & tok = generated_tokens[i];
            f << "  {\"step\": " << i
              << ", \"token_id\": " << tok.token_id
              << ", \"piece\": \"" << json_escape(tok.piece) << "\"}";
            if (i + 1 < generated_tokens.size()) {
                f << ",";
            }
            f << "\n";
        }
        f << "]\n";
    }

    LOG_INF("  saved %d layer + %d router tensors → %s\n",
            n_layer, n_router, st.output_dir.c_str());
}

// ================================================================
// Parse tab-separated prompt file  (id<TAB>text)
// ================================================================
struct prompt_entry { std::string id; std::string text; };

static std::vector<prompt_entry> parse_prompt_file(const std::string & path) {
    std::vector<prompt_entry> out;
    std::ifstream f(path);
    if (!f.is_open()) {
        LOG_ERR("failed to open prompt file: %s\n", path.c_str());
        return out;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) continue;
        auto tab = line.find('\t');
        if (tab == std::string::npos) continue;
        out.push_back({line.substr(0, tab), line.substr(tab + 1)});
    }
    return out;
}

static std::string sanitize_prompt_id(const std::string & raw, size_t fallback_idx) {
    if (raw.empty()) {
        return "prompt_" + std::to_string(fallback_idx + 1);
    }

    std::string out;
    out.reserve(raw.size());
    for (char c : raw) {
        const bool ok = (c >= 'a' && c <= 'z') ||
                        (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') ||
                        c == '_' || c == '-';
        out.push_back(ok ? c : '_');
    }

    if (out.empty()) {
        out = "prompt_" + std::to_string(fallback_idx + 1);
    }
    return out;
}

// ================================================================
// Pre-parse custom CLI flags, strip them from argv
// ================================================================
struct custom_args {
    std::string prompt_file;
    std::string output_dir = "./activations";
    bool        list_tensors = false;
    bool        routing_only = false;
    bool        no_stream    = false;
};

static custom_args strip_custom_args(int & argc, char **& argv) {
    custom_args ca;
    std::vector<char *> keep;
    keep.push_back(argv[0]);

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--prompt-file" && i + 1 < argc) {
            ca.prompt_file = argv[++i];
        } else if ((a == "-o" || a == "--output-dir") && i + 1 < argc) {
            ca.output_dir = argv[++i];
        } else if (a == "--list-tensors") {
            ca.list_tensors = true;
        } else if (a == "--routing-only") {
            ca.routing_only = true;
        } else if (a == "--no-stream") {
            ca.no_stream = true;
        } else {
            keep.push_back(argv[i]);
        }
    }

    // overwrite argv in-place (safe — keep[] points into original argv)
    argc = (int)keep.size();
    for (int i = 0; i < argc; i++) argv[i] = keep[i];
    return ca;
}

// ================================================================
// Main
// ================================================================
int main(int argc, char ** argv) {

    // ---- 1. Extract our flags before llama's parser sees them ----
    custom_args ca = strip_custom_args(argc, argv);

    // ---- 2. Parse standard llama params ----
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    common_init();

    // ---- 3. Set up capture state & callback ----
    capture_state state;
    state.list_tensors_mode = ca.list_tensors;
    state.routing_only      = ca.routing_only;
    state.verbose           = (params.verbosity > 0);

    params.cb_eval           = capture_cb;
    params.cb_eval_user_data = &state;
    params.warmup            = false;

    LOG_INF("=== Consciousness Metrics — Activation Capture ===\n");
    LOG_INF("Output dir : %s\n", ca.output_dir.c_str());
    LOG_INF("n_predict  : %d\n", params.n_predict);
    LOG_INF("Text output: %s\n", ca.no_stream ? "file only (--no-stream)" : "stdout + file");
    if (ca.list_tensors)  LOG_INF("Mode       : --list-tensors (discovery)\n");
    if (ca.routing_only)  LOG_INF("Mode       : --routing-only (router/gate tensors only)\n");

    // ---- 4. Init backend & load model (one-time cost) ----
    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    auto * model    = llama_init->model();
    auto * ctx      = llama_init->context();
    if (!model || !ctx) { LOG_ERR("model/context init failed\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool    add_bos = llama_vocab_get_add_bos(vocab);
    const int32_t eos_id  = llama_vocab_eos(vocab);
    const int     n_vocab  = llama_vocab_n_tokens(vocab);
    common_sampler * smpl  = nullptr;

    if (params.n_predict > 0 && !ca.list_tensors) {
        smpl = common_sampler_init(model, params.sampling);
        if (!smpl) {
            LOG_ERR("failed to initialize sampler\n");
            return 1;
        }
        LOG_INF("Sampler    : %s\n", common_sampler_print(smpl).c_str());
    }

    LOG_INF("EOS token id: %d\n", eos_id);

    // Log all EOG tokens for diagnostic purposes (R1 has multiple)
    LOG_INF("EOG tokens in vocab:\n");
    for (int i = 0; i < n_vocab; i++) {
        if (llama_vocab_is_eog(vocab, i)) {
            char buf[256];
            int n = llama_token_to_piece(vocab, i, buf, sizeof(buf), 0, true);
            std::string tok_str(buf, n > 0 ? n : 0);
            LOG_INF("  token %d = '%s'%s\n", i, tok_str.c_str(),
                    i == eos_id ? " [PRIMARY EOS - will stop generation]" : " [EOG - will NOT stop generation]");
        }
    }

    LOG_INF("%s\n\n", common_params_get_system_info(params).c_str());

    // ---- 5. Build prompt list ----
    std::vector<prompt_entry> prompts;
    if (!ca.prompt_file.empty()) {
        prompts = parse_prompt_file(ca.prompt_file);
        if (prompts.empty()) {
            LOG_ERR("No valid prompts loaded from --prompt-file %s\n", ca.prompt_file.c_str());
            return 1;
        }
    } else if (!params.prompt.empty()) {
        prompts.push_back({"single", params.prompt});
    } else {
        LOG_ERR("No prompts.  Use --prompt-file <tsv> or -p <text>\n");
        return 1;
    }
    LOG_INF("Loaded %zu prompts\n\n", prompts.size());

    // ---- 6. Process each prompt ----
    std::set<std::string> used_prompt_dirs;
    for (size_t pi = 0; pi < prompts.size(); pi++) {
        const auto & pe = prompts[pi];
        const std::string safe_id_base = sanitize_prompt_id(pe.id, pi);
        std::string safe_id = safe_id_base;
        int suffix = 2;
        while (!used_prompt_dirs.insert(safe_id).second) {
            safe_id = safe_id_base + "_" + std::to_string(suffix++);
        }
        if (safe_id_base != pe.id || safe_id != safe_id_base) {
            LOG_INF("  sanitized prompt id '%s' -> '%s'\n", pe.id.c_str(), safe_id.c_str());
        }

        LOG_INF("[%zu/%zu] %s : %.70s%s\n",
                pi + 1, prompts.size(), pe.id.c_str(),
                pe.text.c_str(), pe.text.size() > 70 ? "..." : "");

        // reset per-prompt state
        state.tensors.clear();
        state.discovered_tensor_names.clear();
        state.logged_shape_mismatch.clear();
        state.tensors_seen     = 0;
        state.tensors_captured = 0;
        state.prompt_has_error = false;
        state.current_prompt_id = pe.id;
        state.output_dir        = (fs::path(ca.output_dir) / safe_id).string();
        fs::create_directories(state.output_dir);

        // clear KV cache from previous prompt
        llama_memory_clear(llama_get_memory(ctx), true);
        if (smpl) {
            common_sampler_reset(smpl);
        }

        // tokenize
        std::vector<llama_token> tokens = common_tokenize(ctx, pe.text, add_bos);
        int n_prompt = (int)tokens.size();
        if (tokens.empty()) { LOG_ERR("  empty tokenization — skipping\n"); continue; }

        auto t0 = std::chrono::high_resolution_clock::now();

        // --- prefill (process all prompt tokens) ---
        const int prefill_batch = params.n_batch > 0 ? params.n_batch : 512;
        for (int pos = 0; pos < n_prompt; ) {
            int n_cur = std::min(prefill_batch, n_prompt - pos);
            if (llama_decode(ctx, llama_batch_get_one(tokens.data() + pos, n_cur))) {
                LOG_ERR("  prefill decode failed at token %d/%d — skipping\n", pos, n_prompt);
                state.prompt_has_error = true;
                break;
            }
            if (state.prompt_has_error) {
                break;
            }
            pos += n_cur;
        }

        if (state.prompt_has_error) {
            LOG_ERR("  prompt capture failed during prefill; removing partial output\n");
            fs::remove_all(state.output_dir);
            continue;
        }

        // --- generation with text capture ---
        int n_gen = 0;
        std::string gen_text;
        std::vector<generated_token_info> generated_tokens;

        if (params.n_predict > 0 && !ca.list_tensors) {
            for (int g = 0; g < params.n_predict; g++) {
                llama_token next = common_sampler_sample(smpl, ctx, -1);
                // Only stop on primary EOS token (<｜end▁of▁sentence｜>).
                // Do NOT use llama_vocab_is_eog() — it catches </think>
                // which is marked as EOG in R1's GGUF metadata. We want
                // generation to continue through </think> to capture
                // both the thinking phase AND the visible response.
                if (next == eos_id) break;

                // token is selected — count it and capture text before decode
                n_gen++;
                {
                    char buf[256];
                    int n = llama_token_to_piece(vocab, next, buf, sizeof(buf), 0, true);
                    std::string piece = n > 0 ? std::string(buf, n) : std::string();
                    generated_tokens.push_back({ next, piece });
                    if (n > 0) {
                        gen_text.append(piece);
                        // stream to stdout unless suppressed
                        if (!ca.no_stream) {
                            fwrite(buf, 1, n, stdout);
                            fflush(stdout);
                        }
                    }
                }
                common_sampler_accept(smpl, next, true);

                if (llama_decode(ctx, llama_batch_get_one(&next, 1))) {
                    LOG_ERR("  gen decode failed at token %d\n", g);
                    state.prompt_has_error = true;
                    break;
                }
                if (state.prompt_has_error) {
                    LOG_ERR("  prompt capture failed during generation; aborting prompt\n");
                    break;
                }
            }
        }

        // newline after streamed text
        if (!ca.no_stream && n_gen > 0) {
            fprintf(stdout, "\n");
            fflush(stdout);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        LOG_INF("  tokens: %d prompt + %d gen | tensors: %d seen, %d captured, %zu unique | %.0f ms\n",
                n_prompt, n_gen,
                state.tensors_seen, state.tensors_captured, state.tensors.size(),
                ms);

        if (ca.list_tensors) {
            LOG_INF("\nDiscovered %zu unique tensor names.  Exiting.\n", state.discovered_tensor_names.size());
            break;
        }

        if (state.prompt_has_error) {
            LOG_ERR("  prompt capture failed; removing partial output\n");
            fs::remove_all(state.output_dir);
            continue;
        }

        save_prompt(state, pe.text, gen_text, generated_tokens, n_prompt, n_gen, ms);
    }

    common_sampler_free(smpl);

    // ---- 7. Summary ----
    LOG_INF("\n=== Capture Complete ===\n");
    LOG_INF("Prompts: %zu  |  Output: %s\n", prompts.size(), ca.output_dir.c_str());
    llama_perf_context_print(ctx);
    llama_backend_free();
    return 0;
}
