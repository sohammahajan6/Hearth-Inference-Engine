// The reference forward pass on tiny random Llamas: reference ops and dequant
// against hand-computed values, the whole pass against an independent FP64
// implementation, determinism across runs and thread counts, prefill ==
// step-by-step decode, the tensor dump, and the quantized-activation (cpu)
// matmuls on Q8_0 and Q4_0 weights.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "core/clock.h"
#include "core/cpu_features.h"
#include "core/error.h"
#include "core/fp16.h"
#include "core/log.h"
#include "core/platform.h"
#include "core/quant.h"
#include "core/thread_pool.h"
#include "core/trace.h"
#include "cpu/ref_ops.h"
#include "gguf/gguf.h"
#include "gguf/types.h"
#include "gguf_builder.h"
#include "gpu/context.h"
#include "gpu/hang_ledger.h"
#include "kquant_encode.h"
#include "model/dequant.h"
#include "model/llama.h"
#include "model/llama_gpu.h"
#include "model/llama_ref.h"
#include "test_util.h"

using namespace hearth;

namespace {

std::vector<float> run(const std::vector<std::byte>& bytes, int threads, const std::vector<std::int32_t>& tokens,
                       bool one_by_one, model::Matmul matmul = model::Matmul::kFp32, Isa isa = best_isa(),
                       std::uint32_t n_ctx = 64) {
  const gguf::File f = gguf::File::parse(bytes);
  ThreadPool pool(threads);
  model::LlamaRef llama(f, n_ctx, pool, matmul, isa == Isa::kAvx2 ? Isa::kScalar : isa);
  const std::size_t V = llama.config().n_vocab;
  std::vector<float> logits(tokens.size() * V);
  if (one_by_one) {
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      llama.forward(std::span(tokens).subspan(i, 1), std::span(logits).subspan(i * V, V), false);
    }
  } else {
    llama.forward(tokens, logits, true);
  }
  return logits;
}

// An independent FP64 implementation of the Llama forward pass, written from
// the formulas in docs/architecture/05-forward-pass.md (F32 files only), and
// Qwen2's differences (15-qwen.md): Q/K/V biases when present, NEOX pairs
// (i, i + hd/2) when the config says so, an untied output.weight. It shares
// nothing with cpu-ref except the config parser.
std::vector<double> naive_logits(const gguf::File& f, const std::vector<std::int32_t>& tokens) {
  const model::LlamaConfig c = model::LlamaConfig::from_gguf(f);
  const auto tensor = [&](const std::string& name) {
    const gguf::TensorInfo* t = f.find_tensor(name);
    REQUIRE(t != nullptr);
    const std::span<const std::byte> b = f.tensor_bytes(*t);
    std::vector<double> v(b.size() / 4);
    for (std::size_t i = 0; i < v.size(); ++i) {
      float x;
      std::memcpy(&x, b.data() + 4 * i, 4);
      v[i] = x;
    }
    return v;
  };
  const std::size_t E = c.n_embd, hd = c.head_dim, nh = c.n_head, nkv = c.n_head_kv, F = c.n_ff, V = c.n_vocab;
  const std::size_t T = tokens.size();
  const auto matvec = [](const std::vector<double>& w, const std::vector<double>& x, std::size_t out) {
    std::vector<double> y(out, 0.0);
    for (std::size_t o = 0; o < out; ++o) {
      for (std::size_t i = 0; i < x.size(); ++i) y[o] += w[o * x.size() + i] * x[i];
    }
    return y;
  };
  const auto rmsnorm = [&](const std::vector<double>& x, const std::vector<double>& w) {
    double ss = 0;
    for (double v : x) ss += v * v;
    const double inv = 1.0 / std::sqrt(ss / static_cast<double>(x.size()) + c.rms_eps);
    std::vector<double> y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = x[i] * inv * w[i];
    return y;
  };
  const std::vector<double> embd = tensor("token_embd.weight");
  const std::vector<double> factors =
      f.find_tensor("rope_freqs.weight") != nullptr ? tensor("rope_freqs.weight") : std::vector<double>(hd / 2, 1.0);
  const auto rope = [&](std::vector<double>& v, std::size_t heads, std::size_t pos) {
    for (std::size_t h = 0; h < heads; ++h) {
      for (std::size_t i = 0; i < hd / 2; ++i) {
        const double a = static_cast<double>(pos) *
                         std::pow(static_cast<double>(c.rope_theta), -2.0 * static_cast<double>(i) / hd) / factors[i];
        const std::size_t i0 = c.rope_neox ? i : 2 * i, i1 = c.rope_neox ? i + hd / 2 : 2 * i + 1;
        const double x0 = v[h * hd + i0], x1 = v[h * hd + i1];
        v[h * hd + i0] = x0 * std::cos(a) - x1 * std::sin(a);
        v[h * hd + i1] = x0 * std::sin(a) + x1 * std::cos(a);
      }
    }
  };
  const auto add_bias = [&](std::vector<double>& v, const std::string& name) {
    if (f.find_tensor(name) == nullptr) return;
    const auto bias = tensor(name);
    for (std::size_t i = 0; i < v.size(); ++i) v[i] += bias[i];
  };
  std::vector<std::vector<double>> x(T);
  for (std::size_t t = 0; t < T; ++t) {
    const auto row = embd.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(tokens[t]) * E);
    x[t].assign(row, row + static_cast<std::ptrdiff_t>(E));
  }
  for (std::uint32_t l = 0; l < c.n_layer; ++l) {
    const std::string p = "blk." + std::to_string(l) + ".";
    const auto an = tensor(p + "attn_norm.weight"), wq = tensor(p + "attn_q.weight"), wk = tensor(p + "attn_k.weight"),
               wv = tensor(p + "attn_v.weight"), wo = tensor(p + "attn_output.weight"),
               fn = tensor(p + "ffn_norm.weight"), wg = tensor(p + "ffn_gate.weight"), wu = tensor(p + "ffn_up.weight"),
               wd = tensor(p + "ffn_down.weight");
    std::vector<std::vector<double>> q(T), k(T), v(T);
    for (std::size_t t = 0; t < T; ++t) {
      const auto h = rmsnorm(x[t], an);
      q[t] = matvec(wq, h, nh * hd);
      k[t] = matvec(wk, h, nkv * hd);
      v[t] = matvec(wv, h, nkv * hd);
      add_bias(q[t], p + "attn_q.bias");
      add_bias(k[t], p + "attn_k.bias");
      add_bias(v[t], p + "attn_v.bias");
      rope(q[t], nh, t);
      rope(k[t], nkv, t);
    }
    for (std::size_t t = 0; t < T; ++t) {
      std::vector<double> o(nh * hd, 0.0);
      for (std::size_t head = 0; head < nh; ++head) {
        const std::size_t g = head / (nh / nkv);
        std::vector<double> s(t + 1);
        double mx = -1e300;
        for (std::size_t u = 0; u <= t; ++u) {
          double d = 0;
          for (std::size_t i = 0; i < hd; ++i) d += q[t][head * hd + i] * k[u][g * hd + i];
          s[u] = d / std::sqrt(static_cast<double>(hd));
          mx = std::max(mx, s[u]);
        }
        double sum = 0;
        for (double& e : s) sum += (e = std::exp(e - mx));
        for (std::size_t u = 0; u <= t; ++u) {
          for (std::size_t i = 0; i < hd; ++i) o[head * hd + i] += s[u] / sum * v[u][g * hd + i];
        }
      }
      const auto attn = matvec(wo, o, E);
      for (std::size_t i = 0; i < E; ++i) x[t][i] += attn[i];
      const auto h = rmsnorm(x[t], fn);
      const auto gate = matvec(wg, h, F), up = matvec(wu, h, F);
      std::vector<double> a(F);
      for (std::size_t i = 0; i < F; ++i) a[i] = gate[i] / (1.0 + std::exp(-gate[i])) * up[i];
      const auto down = matvec(wd, a, E);
      for (std::size_t i = 0; i < E; ++i) x[t][i] += down[i];
    }
  }
  const auto on = tensor("output_norm.weight");
  const std::vector<double> head = f.find_tensor("output.weight") != nullptr ? tensor("output.weight") : embd;
  std::vector<double> logits(T * V);
  for (std::size_t t = 0; t < T; ++t) {
    const auto h = rmsnorm(x[t], on);
    const auto row = matvec(head, h, V);  // tied embeddings unless output.weight exists
    std::copy(row.begin(), row.end(), logits.begin() + static_cast<std::ptrdiff_t>(t * V));
  }
  return logits;
}

// Byte offset of tensor `name`'s data inside `bytes` (the parser keeps a view).
std::size_t data_offset(const std::vector<std::byte>& bytes, const char* name) {
  const gguf::File f = gguf::File::parse(bytes);
  const gguf::TensorInfo* t = f.find_tensor(name);
  REQUIRE(t != nullptr);
  return static_cast<std::size_t>(f.tensor_bytes(*t).data() - bytes.data());
}

void put16(std::vector<std::byte>& b, std::size_t at, std::uint16_t v) { std::memcpy(b.data() + at, &v, 2); }

// ggml's reference Q4_0 weight quantizer (quantize_row_q4_0_ref): the scale
// is the signed value of largest magnitude divided by -8.
void quantize_row_q4_0(const float* x, std::int64_t k, quant::BlockQ4_0* out) {
  for (std::int64_t b = 0; b < k / quant::kBlock; ++b) {
    const float* v = x + b * quant::kBlock;
    float amax = 0.0f, max = 0.0f;
    for (int j = 0; j < quant::kBlock; ++j) {
      if (amax < std::fabs(v[j])) {
        amax = std::fabs(v[j]);
        max = v[j];
      }
    }
    const float d = max / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    out[b].d = fp32_to_fp16(d);
    for (int j = 0; j < quant::kBlock / 2; ++j) {
      const auto lo =
          static_cast<std::uint8_t>(std::min(15, static_cast<int>(static_cast<std::int8_t>(v[j] * id + 8.5f))));
      const auto hi =
          static_cast<std::uint8_t>(std::min(15, static_cast<int>(static_cast<std::int8_t>(v[j + 16] * id + 8.5f))));
      out[b].q[j] = static_cast<std::uint8_t>(lo | hi << 4);
    }
  }
}

// The F32 file `f32` written into `out`, a file of the same tensors whose
// matrices are of `type` (8 Q8_0, 2 Q4_0); 1-D tensors are copied.
std::vector<std::byte> requantized(const std::vector<std::byte>& f32, std::vector<std::byte> out, std::uint32_t type) {
  const gguf::File src = gguf::File::parse(f32);
  const gguf::File dst = gguf::File::parse(out);
  for (const gguf::TensorInfo& t : src.tensors()) {
    const gguf::TensorInfo* d = dst.find_tensor(t.name);
    REQUIRE(d != nullptr);
    const std::span<const std::byte> from = src.tensor_bytes(t);
    std::byte* to = out.data() + (dst.tensor_bytes(*d).data() - out.data());
    if (t.n_dims == 1) {
      std::memcpy(to, from.data(), from.size());
      continue;
    }
    const auto k = static_cast<std::int64_t>(t.ne[0]);
    std::vector<float> row(static_cast<std::size_t>(k));
    for (std::int64_t r = 0; r < static_cast<std::int64_t>(t.ne[1]); ++r) {
      std::memcpy(row.data(), from.data() + r * k * 4, static_cast<std::size_t>(k) * 4);
      if (type == 8) {
        quant::quantize_row_q8_0(row.data(), k, reinterpret_cast<quant::BlockQ8_0*>(to) + r * (k / quant::kBlock));
      } else {
        quantize_row_q4_0(row.data(), k, reinterpret_cast<quant::BlockQ4_0*>(to) + r * (k / quant::kBlock));
      }
    }
  }
  return out;
}

// Step W3: a tiny model (n_embd 256, n_ff 512, so every format's blocks fit)
// whose attn_q, attn_k, attn_output, ffn_gate and ffn_up are of `type`, and
// attn_v, ffn_down and token_embd Q8_0 (so the embeddings, which are decoded
// from the file and never converted, and the tied LM head are the same in
// as_q8_0's copy). Pseudo-random blocks with finite, small fp16 scales;
// norms and RoPE factors from `f32`.
std::vector<int> fp16_offsets(std::uint32_t type) {  // ggml-common.h
  switch (type) {
    case 8:
    case 6:
      return {0};  // Q8_0, Q5_0
    case 3:
    case 7:
    case 12:
    case 13:
      return {0, 2};  // Q4_1, Q5_1, Q4_K, Q5_K: d, then m or dmin
    case 10:
      return {80, 82};  // Q2_K
    case 11:
      return {108};  // Q3_K
    case 14:
      return {208};  // Q6_K
    default:
      FAIL("no block layout for type " << type);
      return {};
  }
}

std::vector<std::byte> random_quant_model(const std::vector<std::byte>& f32, std::uint32_t type, std::uint32_t seed) {
  std::vector<std::byte> out = test::tiny_llama(1, 4, 2, type, 256, 128, test::tiny_vocab(), 512, 8).build();
  const gguf::File src = gguf::File::parse(f32);
  const gguf::File dst = gguf::File::parse(out);
  std::uint32_t st = seed;
  const auto next = [&] {
    st = st * 1664525u + 1013904223u;
    return st >> 8;
  };
  for (const gguf::TensorInfo& t : src.tensors()) {
    const gguf::TensorInfo* d = dst.find_tensor(t.name);
    REQUIRE(d != nullptr);
    std::byte* to = out.data() + (dst.tensor_bytes(*d).data() - out.data());
    if (t.n_dims == 1) {
      const std::span<const std::byte> from = src.tensor_bytes(t);
      std::memcpy(to, from.data(), from.size());
      continue;
    }
    const std::vector<int> halves = fp16_offsets(d->type);
    const std::size_t blocks = dst.tensor_bytes(*d).size() / d->traits->type_size;
    for (std::size_t b = 0; b < blocks; ++b) {
      std::byte* blk = to + b * d->traits->type_size;
      for (std::uint32_t i = 0; i < d->traits->type_size; ++i) blk[i] = static_cast<std::byte>(next());
      for (const int off : halves) {
        const std::uint32_t r = next();  // 2^-12 .. 2^-6, either sign
        const auto h =
            static_cast<std::uint16_t>(((r & 1u) << 15) | ((3u + (r >> 1) % 6u) << 10) | ((r >> 5) & 0x3FFu));
        std::memcpy(blk + off, &h, 2);
      }
    }
  }
  return out;
}

// The model `src` with every 2-D tensor decoded and re-quantized to Q8_0
// with ggml's rule: what LlamaRef converts such matrices to at load.
std::vector<std::byte> as_q8_0(const std::vector<std::byte>& src_bytes) {
  std::vector<std::byte> out = test::tiny_llama(1, 4, 2, 8, 256, 128, test::tiny_vocab(), 512).build();
  const gguf::File src = gguf::File::parse(src_bytes);
  const gguf::File dst = gguf::File::parse(out);
  for (const gguf::TensorInfo& t : src.tensors()) {
    const gguf::TensorInfo* d = dst.find_tensor(t.name);
    REQUIRE(d != nullptr);
    const std::span<const std::byte> from = src.tensor_bytes(t);
    std::byte* to = out.data() + (dst.tensor_bytes(*d).data() - out.data());
    if (t.n_dims == 1) {
      std::memcpy(to, from.data(), from.size());
      continue;
    }
    if (t.type == 8) {  // already Q8_0: kept as it is, as the engine imports it
      std::memcpy(to, from.data(), from.size());
      continue;
    }
    const auto k = static_cast<std::int64_t>(t.ne[0]);
    std::vector<float> row(static_cast<std::size_t>(k));
    for (std::int64_t r = 0; r < static_cast<std::int64_t>(t.ne[1]); ++r) {
      model::dequant_row(t, from, r, row.data());
      quant::quantize_row_q8_0(row.data(), k, reinterpret_cast<quant::BlockQ8_0*>(to) + r * (k / quant::kBlock));
    }
  }
  return out;
}

// The tiny model `f32` (built with these shape arguments) with every 2-D tensor
// quantized to `type` (8 Q8_0, 2 Q4_0); norms and RoPE factors stay F32.
std::vector<std::byte> quantized_copy(const std::vector<std::byte>& f32, std::uint32_t n_layer, std::uint32_t n_head,
                                      std::uint32_t n_head_kv, std::uint32_t type, std::uint64_t n_embd = 64) {
  return requantized(f32, test::tiny_llama(n_layer, n_head, n_head_kv, type, n_embd).build(), type);
}

// The tiny model `f32` (built as tiny_llama(n_layer, n_head, n_head_kv, 0,
// n_embd, 128, tiny_vocab(), n_ff)) with every 2-D tensor encoded as `type`
// (12 Q4_K, 14 Q6_K), and attn_v, ffn_down and token_embd as `second` if not 0.
std::vector<std::byte> k_quantized_copy(const std::vector<std::byte>& f32, std::uint32_t n_layer, std::uint32_t n_head,
                                        std::uint32_t n_head_kv, std::uint32_t type, std::uint32_t second,
                                        std::uint64_t n_embd, std::uint64_t n_ff) {
  std::vector<std::byte> out =
      test::tiny_llama(n_layer, n_head, n_head_kv, type, n_embd, 128, test::tiny_vocab(), n_ff, second).build();
  const gguf::File src = gguf::File::parse(f32);
  const gguf::File dst = gguf::File::parse(out);
  for (const gguf::TensorInfo& t : src.tensors()) {
    const gguf::TensorInfo* d = dst.find_tensor(t.name);
    REQUIRE(d != nullptr);
    const std::span<const std::byte> from = src.tensor_bytes(t);
    std::byte* to = out.data() + (dst.tensor_bytes(*d).data() - out.data());
    if (t.n_dims == 1) {
      std::memcpy(to, from.data(), from.size());
      continue;
    }
    const std::int64_t n = static_cast<std::int64_t>(t.ne[0] * t.ne[1]);
    std::vector<float> v(static_cast<std::size_t>(n));
    std::memcpy(v.data(), from.data(), v.size() * 4);
    for (std::int64_t b = 0; b < n / quant::kSuperBlock; ++b) {
      if (d->type == 12) {
        quant::BlockQ4_K blk;
        test::encode_q4_k(v.data() + b * quant::kSuperBlock, blk);
        std::memcpy(to + b * static_cast<std::int64_t>(sizeof(blk)), &blk, sizeof(blk));
      } else {
        REQUIRE(d->type == 14);
        quant::BlockQ6_K blk;
        test::encode_q6_k(v.data() + b * quant::kSuperBlock, blk);
        std::memcpy(to + b * static_cast<std::int64_t>(sizeof(blk)), &blk, sizeof(blk));
      }
    }
  }
  return out;
}

double rel_l2(const std::vector<float>& a, const std::vector<float>& ref) {
  double num = 0, den = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    num += (static_cast<double>(a[i]) - ref[i]) * (static_cast<double>(a[i]) - ref[i]);
    den += static_cast<double>(ref[i]) * ref[i];
  }
  return std::sqrt(num / den);
}

// ACCEPTANCE "Quantised models" statistics of `ours` against `ref`, both
// [positions][V] logits, computed in double as tools/layer_diff.py does:
// KLD(p_ref || p_ours) per position, and the top-1 disagreements that are
// not near-ties (the reference's top-2 probability gap > 0.2).
struct LogitStats {
  double mean_kld = 0, max_kld = 0;
  std::size_t positions = 0, disagreements = 0, non_tie_disagreements = 0;
};

LogitStats logit_stats(const std::vector<float>& ours, const std::vector<float>& ref, std::size_t V) {
  LogitStats s;
  s.positions = ref.size() / V;
  auto log_softmax = [V](const float* x, std::vector<double>& out) {
    double m = x[0];
    for (std::size_t i = 1; i < V; ++i) m = std::max(m, static_cast<double>(x[i]));
    double sum = 0;
    for (std::size_t i = 0; i < V; ++i) sum += std::exp(static_cast<double>(x[i]) - m);
    const double lse = m + std::log(sum);
    out.resize(V);
    for (std::size_t i = 0; i < V; ++i) out[i] = static_cast<double>(x[i]) - lse;
  };
  std::vector<double> lp, lq;
  for (std::size_t p = 0; p < s.positions; ++p) {
    log_softmax(ref.data() + p * V, lp);
    log_softmax(ours.data() + p * V, lq);
    double kld = 0;
    std::size_t top_ref = 0, top_ours = 0;
    for (std::size_t i = 0; i < V; ++i) {
      kld += std::exp(lp[i]) * (lp[i] - lq[i]);
      if (lp[i] > lp[top_ref]) top_ref = i;
      if (lq[i] > lq[top_ours]) top_ours = i;
    }
    s.mean_kld += kld;
    s.max_kld = std::max(s.max_kld, kld);
    if (top_ref != top_ours) {
      ++s.disagreements;
      double second = 0;
      for (std::size_t i = 0; i < V; ++i) {
        if (i != top_ref) second = std::max(second, std::exp(lp[i]));
      }
      if (std::exp(lp[top_ref]) - second > 0.2) ++s.non_tie_disagreements;
    }
  }
  s.mean_kld /= static_cast<double>(s.positions);
  return s;
}

}  // namespace

TEST_SUITE("quick") {
  TEST_CASE("ref ops: rmsnorm, softmax, silu and RoPE match FP64 formulas") {
    const float x[5] = {1.5f, -2.0f, 0.25f, 3.0f, -0.5f};
    const float w[5] = {1.0f, 0.5f, 2.0f, 1.0f, 1.5f};
    float y[5];
    cpu::ref::rmsnorm(x, w, y, 5, 1e-5f);
    double ss = 0;
    for (float v : x) ss += static_cast<double>(v) * v;
    const double inv = 1.0 / std::sqrt(ss / 5 + 1e-5);
    for (int i = 0; i < 5; ++i)
      CHECK(std::fabs(y[i] - x[i] * inv * w[i]) <= 1e-5 * std::fabs(x[i] * inv * w[i]) + 1e-6);

    float s[4] = {1.0f, 2.0f, 3.0f, -1.0f};
    cpu::ref::softmax(s, 4);
    const double den = std::exp(1.0) + std::exp(2.0) + std::exp(3.0) + std::exp(-1.0);
    CHECK(std::fabs(s[2] - std::exp(3.0) / den) <= 1e-6);
    CHECK(std::fabs(s[0] + s[1] + s[2] + s[3] - 1.0f) <= 1e-6);

    CHECK(std::fabs(cpu::ref::silu(1.0f) - 1.0 / (1.0 + std::exp(-1.0))) <= 1e-7);

    // RoPE on one head of width 4 at position 3, theta 100, factors {1, 2}.
    float v[4] = {1.0f, 0.0f, 0.5f, -0.5f};
    const float factors[2] = {1.0f, 2.0f};
    cpu::ref::rope(v, 1, 4, 4, 3, 100.0f, factors);
    const double a0 = 3.0, a1 = 3.0 * std::pow(100.0, -0.5) / 2.0;
    CHECK(std::fabs(v[0] - std::cos(a0)) <= 1e-6);
    CHECK(std::fabs(v[1] - std::sin(a0)) <= 1e-6);
    CHECK(std::fabs(v[2] - (0.5 * std::cos(a1) + 0.5 * std::sin(a1))) <= 1e-6);
    CHECK(std::fabs(v[3] - (0.5 * std::sin(a1) - 0.5 * std::cos(a1))) <= 1e-6);
  }

  TEST_CASE("dequant_row: every supported type decodes rows to hand-computed values") {
    auto bytes = test::GgufBuilder()
                     .tensor({"f32", {32, 2}, 0, 32 * 2 * 4})
                     .tensor({"f16", {32, 2}, 1, 32 * 2 * 2})
                     .tensor({"bf16", {32, 2}, 30, 32 * 2 * 2})
                     .tensor({"q8", {32, 2}, 8, 2 * 34})
                     .tensor({"q4", {32, 2}, 2, 2 * 18})
                     .tensor({"q4k", {256, 1}, 12, 144})
                     .tensor({"q6k", {256, 1}, 14, 210})
                     .tensor({"iq4xs", {256, 1}, 23, 136})
                     .build();
    // F32, row 1: 0.5, -3.25.
    const std::size_t f32 = data_offset(bytes, "f32");
    const float a = 0.5f, b = -3.25f;
    std::memcpy(bytes.data() + f32 + 128, &a, 4);
    std::memcpy(bytes.data() + f32 + 132, &b, 4);
    // F16, row 0: 1, -2, 65504 (largest), 2^-24 (smallest subnormal), -0, +inf; row 1: 0.5.
    const std::size_t f16 = data_offset(bytes, "f16");
    const std::uint16_t h0[6] = {0x3C00, 0xC000, 0x7BFF, 0x0001, 0x8000, 0x7C00};
    for (int j = 0; j < 6; ++j) put16(bytes, f16 + 2 * j, h0[j]);
    put16(bytes, f16 + 64, 0x3800);
    // BF16, row 0: 1, 3.140625, -123.5; row 1, last element: 2^-126 (smallest normal).
    const std::size_t bf = data_offset(bytes, "bf16");
    put16(bytes, bf + 0, 0x3F80);
    put16(bytes, bf + 2, 0x4049);
    put16(bytes, bf + 4, 0xC2F7);
    put16(bytes, bf + 64 + 62, 0x0080);
    // Q8_0: row 0 scale 0.5 with q = j - 16; row 1 scale 1 with q alternating -128, 127.
    const std::size_t q8 = data_offset(bytes, "q8");
    put16(bytes, q8, 0x3800);
    put16(bytes, q8 + 34, 0x3C00);
    for (int j = 0; j < 32; ++j) {
      bytes[q8 + 2 + j] = static_cast<std::byte>(static_cast<std::int8_t>(j - 16));
      bytes[q8 + 34 + 2 + j] = static_cast<std::byte>(static_cast<std::int8_t>((j % 2 != 0) ? 127 : -128));
    }
    // Q4_0: row 0 scale 0.25, byte i = i | (15 - i) << 4, so element i is
    // (i - 8) * 0.25 and element i + 16 is (7 - i) * 0.25; row 1 scale -1 with
    // every byte 0xF0: low nibbles give 8, high nibbles -7.
    const std::size_t q4 = data_offset(bytes, "q4");
    put16(bytes, q4, 0x3400);
    put16(bytes, q4 + 18, 0xBC00);
    for (int i = 0; i < 16; ++i) {
      bytes[q4 + 2 + i] = static_cast<std::byte>(i | (15 - i) << 4);
      bytes[q4 + 18 + 2 + i] = std::byte{0xF0};
    }
    // Q4_K: d = 1, dmin = 0.5. Sub-block 0 has scale 2 and min 3, sub-block 5
    // scale 37 and min 50 (6-bit values in the split encoding of
    // get_scale_min_k4); the others 0. Values 0..31 are the low nibbles of
    // qs[0..31], set to i % 16; values 160..191 the high nibbles of
    // qs[64..95], set to 15 - i % 16.
    const std::size_t q4k = data_offset(bytes, "q4k");
    put16(bytes, q4k, 0x3C00);
    put16(bytes, q4k + 2, 0x3800);
    bytes[q4k + 4 + 0] = std::byte{2};     // sc_0
    bytes[q4k + 4 + 4] = std::byte{3};     // m_0
    bytes[q4k + 4 + 9] = std::byte{0x25};  // sc_5 & 15 = 5, m_5 & 15 = 2
    bytes[q4k + 4 + 1] = std::byte{0x80};  // sc_5 >> 4 = 2, in the top bits of byte 1
    bytes[q4k + 4 + 5] = std::byte{0xC0};  // m_5 >> 4 = 3, in the top bits of byte 5
    for (int i = 0; i < 32; ++i) {
      bytes[q4k + 16 + i] = static_cast<std::byte>(i % 16);
      bytes[q4k + 16 + 64 + i] = static_cast<std::byte>((15 - i % 16) << 4);
    }
    // Q6_K: d = 0.5; unit 0 scale -3, unit 15 scale 127, others 0. Value 0 has
    // code 63 (ql[0] low nibble 15, qh[0] bits 0-1 = 3); value 255 code 42
    // (ql[127] high nibble 10, qh[63] bits 6-7 = 2); every other code is 0.
    const std::size_t q6k = data_offset(bytes, "q6k");
    put16(bytes, q6k + 208, 0x3800);
    bytes[q6k + 192 + 0] = static_cast<std::byte>(static_cast<std::int8_t>(-3));
    bytes[q6k + 192 + 15] = std::byte{127};
    bytes[q6k + 0] = std::byte{0x0F};
    bytes[q6k + 128 + 0] = std::byte{0x03};
    bytes[q6k + 127] = std::byte{0xA0};
    bytes[q6k + 128 + 63] = std::byte{0x80};

    const gguf::File f = gguf::File::parse(bytes);
    std::vector<float> out(256);
    const auto row = [&](const char* name, std::int64_t r) {
      const gguf::TensorInfo* t = f.find_tensor(name);
      model::dequant_row(*t, f.tensor_bytes(*t), r, out.data());
      return out;
    };
    auto r = row("f32", 0);
    CHECK(r[0] == 0.0f);
    r = row("f32", 1);
    CHECK(r[0] == 0.5f);
    CHECK(r[1] == -3.25f);
    r = row("f16", 0);
    CHECK(r[0] == 1.0f);
    CHECK(r[1] == -2.0f);
    CHECK(r[2] == 65504.0f);
    CHECK(r[3] == std::ldexp(1.0f, -24));
    CHECK((r[4] == 0.0f && std::signbit(r[4])));
    CHECK(std::isinf(r[5]));
    CHECK(row("f16", 1)[0] == 0.5f);
    r = row("bf16", 0);
    CHECK(r[0] == 1.0f);
    CHECK(r[1] == 3.140625f);
    CHECK(r[2] == -123.5f);
    CHECK(row("bf16", 1)[31] == std::ldexp(1.0f, -126));
    r = row("q8", 0);
    for (int j = 0; j < 32; ++j) CHECK(r[j] == 0.5f * static_cast<float>(j - 16));
    r = row("q8", 1);
    for (int j = 0; j < 32; ++j) CHECK(r[j] == ((j % 2 != 0) ? 127.0f : -128.0f));
    r = row("q4", 0);
    for (int i = 0; i < 16; ++i) {
      CHECK(r[i] == 0.25f * static_cast<float>(i - 8));
      CHECK(r[i + 16] == 0.25f * static_cast<float>(7 - i));
    }
    r = row("q4", 1);
    for (int i = 0; i < 16; ++i) {
      CHECK(r[i] == 8.0f);
      CHECK(r[i + 16] == -7.0f);
    }
    r = row("q4k", 0);
    for (int i = 0; i < 32; ++i) {
      CHECK(r[i] == 2.0f * static_cast<float>(i % 16) - 1.5f);               // (1 * 2) * q - 0.5 * 3
      CHECK(r[160 + i] == 37.0f * static_cast<float>(15 - i % 16) - 25.0f);  // (1 * 37) * q - 0.5 * 50
      CHECK(r[128 + i] == 0.0f);
    }
    r = row("q6k", 0);
    CHECK(r[0] == -46.5f);                              // (0.5 * -3) * (63 - 32)
    for (int i = 1; i < 16; ++i) CHECK(r[i] == 48.0f);  // (0.5 * -3) * (0 - 32)
    for (int i = 16; i < 240; ++i) CHECK(r[i] == 0.0f);
    for (int i = 240; i < 255; ++i) CHECK(r[i] == -2032.0f);  // (0.5 * 127) * (0 - 32)
    CHECK(r[255] == 635.0f);                                  // (0.5 * 127) * (42 - 32)
    // A type the engine does not load (Q5_K was one until Step W3; its
    // decoding is checked in tests/kernels/dequant_more_test.cpp).
    const gguf::TensorInfo* k = f.find_tensor("iq4xs");
    CHECK_THROWS_WITH_AS(model::dequant_row(*k, f.tensor_bytes(*k), 0, out.data()),
                         doctest::Contains("cannot dequantize tensor 'iq4xs'"), Error);
  }

  TEST_CASE("cpu-ref and cpu: a Qwen2 model (biases, NEOX RoPE, untied output) matches the FP64 implementation") {
    const auto bytes = test::tiny_qwen2(2, 4, 2).random_f32_data(17).build();
    const gguf::File f = gguf::File::parse(bytes);
    const model::LlamaConfig c = model::LlamaConfig::from_gguf(f);
    CHECK(c.arch == "qwen2");
    CHECK(c.rope_neox);
    CHECK_FALSE(c.tied_embeddings);
    std::vector<std::int32_t> tokens;
    for (int i = 0; i < 40; ++i) tokens.push_back((i * 5 + 2) % 32);
    const std::vector<double> ref = naive_logits(f, tokens);
    const auto got = run(bytes, 4, tokens, false);
    double num = 0, den = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
      num += (got[i] - ref[i]) * (got[i] - ref[i]);
      den += ref[i] * ref[i];
    }
    const std::string msg = std::format("rel-L2 of cpu-ref vs FP64 = {:.3e}", std::sqrt(num / den));
    CHECK_MESSAGE(std::sqrt(num / den) < 1e-5, msg);
    // The cpu backend computes F32 weights as cpu-ref does: the same bits, and prefill = decode.
    CHECK(run(bytes, 4, tokens, false, model::Matmul::kQ8Activations) == got);
    CHECK(run(bytes, 3, tokens, true) == got);
    // The full-GPU backend: "gpu backend: a Qwen2 model ..." below.
  }

  TEST_CASE("cpu-ref: logits match an independent FP64 implementation (2 layers, GQA 4:2, 40 positions)") {
    const auto bytes = test::tiny_llama(2, 4, 2).random_f32_data(11).build();
    const gguf::File f = gguf::File::parse(bytes);
    std::vector<std::int32_t> tokens;
    for (int i = 0; i < 40; ++i) tokens.push_back((i * 7 + 3) % 32);
    const std::vector<double> ref = naive_logits(f, tokens);
    ThreadPool pool(4);
    model::LlamaRef llama(f, 64, pool);
    std::vector<float> got(tokens.size() * 32);
    llama.forward(tokens, got, true);
    double num = 0, den = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
      num += (got[i] - ref[i]) * (got[i] - ref[i]);
      den += ref[i] * ref[i];
    }
    const double rel = std::sqrt(num / den);
    const std::string msg = std::format("logits rel-L2 vs FP64 = {:.3e}", rel);
    CHECK_MESSAGE(rel <= 1e-5, msg);
  }

  // The 3B's shapes: 24 query heads share 8 kv heads (ratio 3, not a power of
  // two) and the width 3072 is not a power of two. Width 96 with 6:2 heads has
  // both properties at test size. (The real head width, 128, is covered by the
  // attention kernel tests and by the 3B gates in bench/oracle/3b/.)
  TEST_CASE("cpu-ref: logits match an independent FP64 implementation (GQA 6:2 like the 3B, width 96)") {
    const auto bytes = test::tiny_llama(2, 6, 2, 0, 96).random_f32_data(13).build();
    const gguf::File f = gguf::File::parse(bytes);
    std::vector<std::int32_t> tokens;
    for (int i = 0; i < 40; ++i) tokens.push_back((i * 11 + 5) % 32);
    const std::vector<double> ref = naive_logits(f, tokens);
    ThreadPool pool(4);
    model::LlamaRef llama(f, 64, pool);
    std::vector<float> got(tokens.size() * 32);
    llama.forward(tokens, got, true);
    double num = 0, den = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
      num += (got[i] - ref[i]) * (got[i] - ref[i]);
      den += ref[i] * ref[i];
    }
    const double rel = std::sqrt(num / den);
    const std::string msg = std::format("logits rel-L2 vs FP64 = {:.3e}", rel);
    CHECK_MESSAGE(rel <= 1e-5, msg);
  }

  TEST_CASE("cpu backend: GQA 6:2 at width 96 is deterministic, and prefill equals decode") {
    const auto f32 = test::tiny_llama(2, 6, 2, 0, 96).random_f32_data(17).build();
    const std::vector<std::int32_t> tokens = {1, 17, 3, 29, 5, 11, 7, 23, 13, 2, 19, 31};
    for (const std::uint32_t type : {8u, 2u}) {
      CAPTURE(type);
      const auto q = quantized_copy(f32, 2, 6, 2, type, 96);
      const auto ref = run(q, 4, tokens, false, model::Matmul::kFp32);
      const auto cpu = run(q, 1, tokens, false, model::Matmul::kQ8Activations);
      const double rel = rel_l2(cpu, ref);
      const std::string msg = std::format("rel-L2 of cpu vs cpu-ref logits = {:.3e}", rel);
      CHECK_MESSAGE((rel > 1e-6 && rel < 3e-2), msg);
      for (int t : {2, 4, 8}) {
        if (t > cpu_features().logical_cores) continue;
        const auto other = run(q, t, tokens, false, model::Matmul::kQ8Activations);
        CHECK(std::memcmp(cpu.data(), other.data(), cpu.size() * sizeof(float)) == 0);
      }
      const auto decode = run(q, 4, tokens, true, model::Matmul::kQ8Activations);
      CHECK(std::memcmp(cpu.data(), decode.data(), cpu.size() * sizeof(float)) == 0);
      const auto scalar = run(q, 3, tokens, false, model::Matmul::kQ8Activations, Isa::kScalar);
      CHECK(std::memcmp(cpu.data(), scalar.data(), cpu.size() * sizeof(float)) == 0);
    }
  }

  TEST_CASE("cpu-ref: logits are bit-identical for 1, 2, 4 and 8 threads") {
    const auto bytes = test::tiny_llama().random_f32_data(42).build();
    const std::vector<std::int32_t> tokens = {1, 5, 9, 2, 31, 7, 0, 16};
    const auto base = run(bytes, 1, tokens, false);
    for (int t : {2, 4, 8}) {
      if (t > cpu_features().logical_cores) continue;
      const auto other = run(bytes, t, tokens, false);
      CHECK(std::memcmp(base.data(), other.data(), base.size() * sizeof(float)) == 0);
    }
    // Sanity: the logits are not degenerate.
    float lo = base[0], hi = base[0];
    for (float v : base) {
      lo = std::fmin(lo, v);
      hi = std::fmax(hi, v);
    }
    CHECK(hi - lo > 1e-3f);
  }

  TEST_CASE("cpu-ref: three runs give bit-identical logits") {
    const auto bytes = test::tiny_llama(2, 4, 2).random_f32_data(5).build();
    const std::vector<std::int32_t> tokens = {4, 8, 15, 16, 23, 31, 0, 1};
    const auto first = run(bytes, 4, tokens, false);
    for (int i = 0; i < 2; ++i) {
      const auto again = run(bytes, 4, tokens, false);
      CHECK(std::memcmp(first.data(), again.data(), first.size() * sizeof(float)) == 0);
    }
  }

  TEST_CASE("cpu-ref: one prefill pass equals decoding the tokens one at a time, bit for bit") {
    const auto bytes = test::tiny_llama().random_f32_data(7).build();
    const std::vector<std::int32_t> tokens = {3, 3, 14, 27, 1, 30, 8, 12, 19, 4};
    const auto prefill = run(bytes, 4, tokens, false);
    const auto decode = run(bytes, 4, tokens, true);
    CHECK(std::memcmp(prefill.data(), decode.data(), prefill.size() * sizeof(float)) == 0);
  }

  TEST_CASE("cpu-ref: a dump holds exactly the requested tensors and positions") {
    const auto bytes = test::tiny_llama(2, 4, 2).random_f32_data(3).build();
    const gguf::File f = gguf::File::parse(bytes);
    ThreadPool pool(2);
    model::LlamaRef llama(f, 64, pool);
    const std::vector<std::int32_t> tokens = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20};
    std::vector<float> logits(tokens.size() * 32);
    model::TensorDump dump({"l_out", "result_output"}, {7, 0, 3});
    llama.forward(tokens, logits, true, &dump);
    const std::filesystem::path dir = test::repo_dir() / "build" / "test-tmp" / "dump";
    std::filesystem::remove_all(dir);
    dump.write(dir, "hearth cpu-ref", "tiny", tokens, 64);
    CHECK(std::filesystem::exists(dir / "index.json"));
    CHECK(std::filesystem::file_size(dir / "l_out-0.bin") == 3 * 64 * 4);
    CHECK(std::filesystem::file_size(dir / "l_out-1.bin") == 3 * 64 * 4);
    CHECK_FALSE(std::filesystem::exists(dir / "attn_norm-0.bin"));
    // result_output: rows for positions 0, 3, 7, in that order, equal to the logits.
    std::vector<float> rows(3 * 32);
    {
      std::ifstream in(dir / "result_output.bin", std::ios::binary);
      in.read(reinterpret_cast<char*>(rows.data()), static_cast<std::streamsize>(rows.size() * sizeof(float)));
      REQUIRE(in);
    }  // closed here: Windows cannot delete an open file
    int k = 0;
    for (int p : {0, 3, 7}) {
      CHECK(std::memcmp(rows.data() + 32 * k++, logits.data() + 32 * p, 32 * sizeof(float)) == 0);
    }
    std::filesystem::remove_all(dir);
  }

  TEST_CASE("cpu backend: Q8_0 and Q4_0 weights take the quantized-activation path, deterministically") {
    const auto f32 = test::tiny_llama(2, 4, 2).random_f32_data(21).build();
    const std::vector<std::int32_t> tokens = {1, 17, 3, 29, 5, 11, 7, 23, 13, 2, 19, 31};
    for (const std::uint32_t type : {8u, 2u}) {
      CAPTURE(type);
      const auto q = quantized_copy(f32, 2, 4, 2, type);
      const auto ref = run(q, 4, tokens, false, model::Matmul::kFp32);
      const auto cpu = run(q, 1, tokens, false, model::Matmul::kQ8Activations);
      // Quantizing the activations to 8 bits moves the logits a little, never a lot.
      const double rel = rel_l2(cpu, ref);
      const std::string msg = std::format("rel-L2 of cpu vs cpu-ref logits = {:.3e}", rel);
      CHECK_MESSAGE((rel > 1e-6 && rel < 3e-2), msg);
      for (int t : {2, 4, 8}) {
        if (t > cpu_features().logical_cores) continue;
        const auto other = run(q, t, tokens, false, model::Matmul::kQ8Activations);
        CHECK(std::memcmp(cpu.data(), other.data(), cpu.size() * sizeof(float)) == 0);
      }
      const auto decode = run(q, 4, tokens, true, model::Matmul::kQ8Activations);
      CHECK(std::memcmp(cpu.data(), decode.data(), cpu.size() * sizeof(float)) == 0);
      // ADR 0009: the scalar and AVX-512 kernels compute the same canonical order.
      const auto scalar = run(q, 3, tokens, false, model::Matmul::kQ8Activations, Isa::kScalar);
      CHECK(std::memcmp(cpu.data(), scalar.data(), cpu.size() * sizeof(float)) == 0);
    }
    // F32 weights have no quantized path: the cpu backend computes them as cpu-ref does.
    const auto a = run(f32, 4, tokens, false, model::Matmul::kFp32);
    const auto b = run(f32, 4, tokens, false, model::Matmul::kQ8Activations);
    CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
  }

  TEST_CASE("cpu backend: Q4_K and Q6_K weights take the K-quant path, deterministically") {
    // Width 256 and feed-forward 512: every matrix row is whole super-blocks.
    // 40 tokens: the products of a prefill use the static row split, those
    // of decode the dynamic one (more than and at most 32 tokens).
    const auto f32 = test::tiny_llama(2, 4, 2, 0, 256, 128, test::tiny_vocab(), 512).random_f32_data(31).build();
    std::vector<std::int32_t> tokens;
    for (int i = 0; i < 40; ++i) tokens.push_back((i * 7 + 3) % 32);
    struct Mix {
      std::uint32_t type, second;
    };
    for (const Mix mix : {Mix{12, 0}, Mix{14, 0}, Mix{12, 14}}) {
      CAPTURE(mix.type);
      CAPTURE(mix.second);
      const auto q = k_quantized_copy(f32, 2, 4, 2, mix.type, mix.second, 256, 512);
      const auto ref = run(q, 4, tokens, false, model::Matmul::kFp32);
      const auto cpu = run(q, 1, tokens, false, model::Matmul::kQ8Activations);
      // Q8_K activations move the logits a little, never a lot; an unchanged
      // result would mean the K-quant path was not taken.
      const double rel = rel_l2(cpu, ref);
      const std::string msg = std::format("rel-L2 of cpu vs cpu-ref logits = {:.3e}", rel);
      CHECK_MESSAGE((rel > 1e-6 && rel < 3e-2), msg);
      for (int t : {2, 4, 8}) {
        if (t > cpu_features().logical_cores) continue;
        const auto other = run(q, t, tokens, false, model::Matmul::kQ8Activations);
        CHECK(std::memcmp(cpu.data(), other.data(), cpu.size() * sizeof(float)) == 0);
      }
      const auto decode = run(q, 4, tokens, true, model::Matmul::kQ8Activations);
      CHECK(std::memcmp(cpu.data(), decode.data(), cpu.size() * sizeof(float)) == 0);
      const auto scalar = run(q, 3, tokens, false, model::Matmul::kQ8Activations, Isa::kScalar);
      CHECK(std::memcmp(cpu.data(), scalar.data(), cpu.size() * sizeof(float)) == 0);
    }
  }

  // The `cpu` backend keeps K in blocks of 16 positions (cpu/attention.h). A
  // context of 37 ends in a partly used block, and filling it completely
  // reaches every position of that block.
  TEST_CASE("cpu backend: attention on the blocked K cache gives cpu-ref's bits at a context of 37") {
    const auto f32 = test::tiny_llama(2, 6, 2, 0, 96).random_f32_data(29).build();
    std::vector<std::int32_t> tokens;
    for (int i = 0; i < 37; ++i) tokens.push_back((i * 13 + 2) % 32);
    for (const Isa isa : {Isa::kScalar, best_isa()}) {
      CAPTURE(to_string(isa));
      const auto ref = run(f32, 4, tokens, false, model::Matmul::kFp32, isa, 37);
      const auto cpu = run(f32, 4, tokens, false, model::Matmul::kQ8Activations, isa, 37);
      CHECK(std::memcmp(ref.data(), cpu.data(), ref.size() * sizeof(float)) == 0);
      const auto decode = run(f32, 3, tokens, true, model::Matmul::kQ8Activations, isa, 37);
      CHECK(std::memcmp(ref.data(), decode.data(), ref.size() * sizeof(float)) == 0);
    }
  }

  // The GPU has kernels for Q8_0 and Q4_0 only: a K-quant file
  // or a full-precision one is refused with a clear message, never run on
  // the CPU in silence.
  // Step W3: formats without kernels of their own are converted to Q8_0 at
  // load, so the cpu backend's logits are exactly those of the same model
  // stored as Q8_0 (as_q8_0), and finite. (Q4_K and Q6_K keep their own
  // kernels on the CPU: "cpu backend: Q4_K and Q6_K weights take ...".)
  TEST_CASE("cpu backend: Q4_1, Q5_0, Q5_1, Q2_K, Q3_K and Q5_K run as converted Q8_0, bit for bit") {
    const auto f32 = test::tiny_llama(1, 4, 2, 0, 256, 128, test::tiny_vocab(), 512).random_f32_data(71).build();
    std::vector<std::int32_t> tokens;
    for (int i = 0; i < 9; ++i) tokens.push_back((i * 7 + 3) % 32);
    for (const std::uint32_t type : {3u, 6u, 7u, 10u, 11u, 13u}) {
      CAPTURE(type);
      const auto src = random_quant_model(f32, type, 100 + type);
      const auto want = run(as_q8_0(src), 2, tokens, false, model::Matmul::kQ8Activations);
      const auto got = run(src, 2, tokens, false, model::Matmul::kQ8Activations);
      REQUIRE(got.size() == want.size());
      CHECK(std::memcmp(got.data(), want.data(), got.size() * sizeof(float)) == 0);
      CHECK(std::all_of(got.begin(), got.end(), [](float v) { return std::isfinite(v); }));
      // Decoding one token at a time gives the same bits (the Q8_0 kernels' rule).
      const auto step = run(src, 3, tokens, true, model::Matmul::kQ8Activations);
      CHECK(std::memcmp(got.data(), step.data(), got.size() * sizeof(float)) == 0);
    }
  }

  // The GPU has kernels for Q8_0 and Q4_0; Q4_1, Q5_0, Q5_1, Q2_K, Q3_K and
  // Q5_K are converted to Q8_0 at load, as on the CPU, so their GPU logits
  // are the GPU's logits for the same model stored as Q8_0. Full precision,
  // Q4_K and Q6_K are refused with a clear message, never run on the CPU in
  // silence.
  TEST_CASE("gpu backend: Q5_0, Q5_K and Q2_K run as converted Q8_0; full precision and Q4_K/Q6_K are refused") {
    if (!gpu::vulkan_available()) return;
    const gpu::ContextOptions opts = test::gpu_test_options();
    ThreadPool pool(2);
    const auto f32 = test::tiny_llama(1, 4, 2, 0, 256, 128, test::tiny_vocab(), 512).random_f32_data(33).build();
    {
      const gguf::File f = gguf::File::parse(f32);
      CHECK_THROWS_WITH_AS(model::LlamaRef(f, 64, pool, model::Matmul::kGpu, Isa::kScalar, &opts),
                           doctest::Contains("of type F32"), Error);
    }
    {
      const auto q = k_quantized_copy(f32, 1, 4, 2, 12, 14, 256, 512);
      const gguf::File f = gguf::File::parse(q);
      CHECK_THROWS_WITH_AS(model::LlamaRef(f, 64, pool, model::Matmul::kGpu, Isa::kScalar, &opts),
                           doctest::Contains("Q4_K and Q6_K run on the CPU only"), Error);
    }
    std::vector<std::int32_t> tokens;
    for (int i = 0; i < 9; ++i) tokens.push_back((i * 5 + 1) % 32);
    const auto gpu_logits = [&](const std::vector<std::byte>& bytes) {
      const gguf::File f = gguf::File::parse(bytes);
      model::LlamaRef llama(f, 64, pool, model::Matmul::kGpu, Isa::kScalar, &opts);
      std::vector<float> logits(tokens.size() * llama.config().n_vocab);
      llama.forward(tokens, logits, true);
      CHECK(llama.gpu_pass()->validation_messages() == 0);
      return logits;
    };
    for (const std::uint32_t type : {6u, 13u, 10u}) {  // Q5_0, Q5_K, Q2_K
      CAPTURE(type);
      const auto src = random_quant_model(f32, type, 200 + type);
      const auto want = gpu_logits(as_q8_0(src));
      const auto got = gpu_logits(src);
      REQUIRE(got.size() == want.size());
      CHECK(std::memcmp(got.data(), want.data(), got.size() * sizeof(float)) == 0);
      CHECK(std::all_of(got.begin(), got.end(), [](float v) { return std::isfinite(v); }));
    }
  }

  // 08 doc step 6.4: the whole forward pass on the GPU. Its matrix products
  // are the CPU's bits, but RMSNorm, SiLU and attention are FP32 on the GPU
  // (the CPU sums in double), so its logits are close to the cpu backend's,
  // not equal. The bars are ACCEPTANCE's for the GPU backend against the CPU
  // path; under 1000 positions, the short-run ones: mean KLD <= 2e-3 and
  // top-1 disagreements only at near-ties. Within the GPU pass, prefill
  // equals decode bit for bit, and a second run gives the same bits.
  TEST_CASE(
      "gpu backend: logits within the quantized bars of the cpu backend's; prefill = decode and reruns bit for "
      "bit (validation on)") {
    if (!gpu::vulkan_available()) return;
    const gpu::ContextOptions opts = test::gpu_test_options();
    for (const auto& [n_head, n_kv, width] :
         {std::tuple<std::uint32_t, std::uint32_t, std::uint64_t>{4, 2, 64}, {6, 2, 96}}) {
      const auto f32 = test::tiny_llama(2, n_head, n_kv, 0, width).random_f32_data(37).build();
      std::vector<std::int32_t> tokens;
      for (int i = 0; i < 70; ++i) tokens.push_back((i * 7 + 3) % 32);  // more than one 64-token GPU chunk
      for (const std::uint32_t type : {8u, 2u}) {
        CAPTURE(n_head);
        CAPTURE(type);
        const auto q = quantized_copy(f32, 2, n_head, n_kv, type, width);
        const gguf::File f = gguf::File::parse(q);
        const auto cpu = run(q, 4, tokens, false, model::Matmul::kQ8Activations, best_isa(), 128);
        const Isa isa = best_isa() == Isa::kAvx2 ? Isa::kScalar : best_isa();
        auto gpu_logits = [&](bool one_by_one) {
          ThreadPool pool(4);
          model::LlamaRef llama(f, 128, pool, model::Matmul::kGpu, isa, &opts);
          const std::size_t V = llama.config().n_vocab;
          std::vector<float> got(tokens.size() * V);
          if (one_by_one) {
            for (std::size_t i = 0; i < tokens.size(); ++i) {
              llama.forward(std::span(tokens).subspan(i, 1), std::span(got).subspan(i * V, V), false);
            }
          } else {
            llama.forward(tokens, got, true);
          }
          REQUIRE(llama.gpu_pass() != nullptr);
          CHECK(llama.gpu_pass()->validation_messages() == 0);
          return got;
        };
        const auto prefill = gpu_logits(false);
        const LogitStats st = logit_stats(prefill, cpu, prefill.size() / tokens.size());
        const std::string msg = std::format(
            "gpu vs cpu: {} positions, mean KLD {:.3e}, max {:.3e}, top-1 disagreements {} ({} not "
            "near-ties); logits rel-L2 {:.3e}",
            st.positions, st.mean_kld, st.max_kld, st.disagreements, st.non_tie_disagreements, rel_l2(prefill, cpu));
        MESSAGE(msg);
        CHECK_MESSAGE(st.mean_kld <= 2e-3, msg);
        CHECK_MESSAGE(st.non_tie_disagreements == 0, msg);
        // Control: the bar must be able to fail on this tiny model. The cpu
        // backend's own logits one position late are plausible but wrong.
        {
          const std::size_t V = prefill.size() / tokens.size();
          const std::vector<float> late(cpu.begin(), cpu.end() - static_cast<std::ptrdiff_t>(V));
          const std::vector<float> ref(cpu.begin() + static_cast<std::ptrdiff_t>(V), cpu.end());
          const LogitStats wrong = logit_stats(late, ref, V);
          const std::string wmsg = std::format("control (one position late): mean KLD {:.3e}", wrong.mean_kld);
          MESSAGE(wmsg);
          CHECK_MESSAGE(wrong.mean_kld > 2e-3, wmsg);
        }
        CHECK(std::memcmp(prefill.data(), gpu_logits(true).data(), prefill.size() * sizeof(float)) == 0);
        CHECK(std::memcmp(prefill.data(), gpu_logits(false).data(), prefill.size() * sizeof(float)) == 0);
      }
    }
  }

  // The whole Qwen2 forward pass on the GPU (15-qwen.md section 4): the
  // Q/K/V biases and NEOX RoPE, against the cpu backend with the bars of the
  // Llama case above. GQA 4:2, and 14:2 (Qwen2.5-0.5B's ratio of 7) at width
  // 224 = 7 blocks, a partial 16-block group as Qwen's 896. Control: the cpu
  // backend on the same file with its biases zeroed is outside the bar, so a
  // GPU pass that skipped them (or added them after RoPE's input was read)
  // would fail; the RoPE pairing itself is checked bit for bit in gpu_ops_test.
  TEST_CASE(
      "gpu backend: a Qwen2 model (biases, NEOX RoPE) within the quantized bars of the cpu backend's; prefill = "
      "decode and reruns bit for bit (validation on)") {
    if (!gpu::vulkan_available()) return;
    const gpu::ContextOptions opts = test::gpu_test_options();
    for (const auto& [n_head, n_kv, width] :
         {std::tuple<std::uint32_t, std::uint32_t, std::uint64_t>{4, 2, 64}, {14, 2, 224}}) {
      const auto f32 = test::tiny_qwen2(2, n_head, n_kv, width).random_f32_data(53).build();
      std::vector<std::int32_t> tokens;
      for (int i = 0; i < 70; ++i) tokens.push_back((i * 5 + 2) % 32);  // more than one 64-token GPU chunk
      for (const std::uint32_t type : {8u, 2u}) {
        CAPTURE(n_head);
        CAPTURE(type);
        const auto q = requantized(f32, test::tiny_qwen2(2, n_head, n_kv, width, 96, type).build(), type);
        const gguf::File f = gguf::File::parse(q);
        REQUIRE(model::LlamaConfig::from_gguf(f).rope_neox);
        const auto cpu = run(q, 4, tokens, false, model::Matmul::kQ8Activations, best_isa(), 128);
        const Isa isa = best_isa() == Isa::kAvx2 ? Isa::kScalar : best_isa();
        auto gpu_logits = [&](bool one_by_one) {
          ThreadPool pool(4);
          model::LlamaRef llama(f, 128, pool, model::Matmul::kGpu, isa, &opts);
          const std::size_t V = llama.config().n_vocab;
          std::vector<float> got(tokens.size() * V);
          if (one_by_one) {
            for (std::size_t i = 0; i < tokens.size(); ++i) {
              llama.forward(std::span(tokens).subspan(i, 1), std::span(got).subspan(i * V, V), false);
            }
          } else {
            llama.forward(tokens, got, true);
          }
          REQUIRE(llama.gpu_pass() != nullptr);
          CHECK(llama.gpu_pass()->validation_messages() == 0);
          return got;
        };
        const auto prefill = gpu_logits(false);
        const std::size_t V = prefill.size() / tokens.size();
        const LogitStats st = logit_stats(prefill, cpu, V);
        const std::string msg = std::format(
            "qwen2 gpu vs cpu: {} positions, mean KLD {:.3e}, max {:.3e}, top-1 disagreements {} ({} not "
            "near-ties); logits rel-L2 {:.3e}",
            st.positions, st.mean_kld, st.max_kld, st.disagreements, st.non_tie_disagreements, rel_l2(prefill, cpu));
        MESSAGE(msg);
        CHECK_MESSAGE(st.mean_kld <= 2e-3, msg);
        CHECK_MESSAGE(st.non_tie_disagreements == 0, msg);
        {
          std::vector<std::byte> no_bias = q;
          const gguf::File nf = gguf::File::parse(no_bias);
          int zeroed = 0;
          for (const gguf::TensorInfo& t : nf.tensors()) {
            if (!t.name.ends_with(".bias")) continue;
            const std::span<const std::byte> b = nf.tensor_bytes(t);
            std::memset(no_bias.data() + (b.data() - no_bias.data()), 0, b.size());
            ++zeroed;
          }
          CHECK(zeroed == 6);  // q, k, v in each of the 2 layers
          const auto cpu0 = run(no_bias, 4, tokens, false, model::Matmul::kQ8Activations, best_isa(), 128);
          const LogitStats wrong = logit_stats(prefill, cpu0, V);
          const std::string wmsg = std::format("control (cpu without biases): mean KLD {:.3e}", wrong.mean_kld);
          MESSAGE(wmsg);
          CHECK_MESSAGE(wrong.mean_kld > 2e-3, wmsg);
        }
        CHECK(std::memcmp(prefill.data(), gpu_logits(true).data(), prefill.size() * sizeof(float)) == 0);
        CHECK(std::memcmp(prefill.data(), gpu_logits(false).data(), prefill.size() * sizeof(float)) == 0);
      }
    }
  }

  // 08 doc section 4 (step 6.5): chunks shrink as attention grows with the
  // position, within [1, min(kChunk, remaining)].
  TEST_CASE("gpu backend: chunk_tokens stays within its bounds and never grows with the position") {
    if (!gpu::vulkan_available()) return;
    const gpu::ContextOptions opts = test::gpu_test_options();
    const auto q = quantized_copy(test::tiny_llama(2, 4, 2).random_f32_data(59).build(), 2, 4, 2, 8);
    const gguf::File f = gguf::File::parse(q);
    ThreadPool pool(2);
    model::LlamaRef llama(f, 64, pool, model::Matmul::kGpu, Isa::kScalar, &opts);
    const model::LlamaGpu& g = *llama.gpu_pass();
    CHECK(g.chunk_tokens(0, 70, false) == model::LlamaGpu::kChunk);  // a tiny model: every chunk is cheap
    CHECK(g.chunk_tokens(0, 5, true) == 5);
    CHECK(g.chunk_tokens(0, 0, false) == 1);
    CHECK(g.chunk_tokens(std::int64_t{1} << 40, 64, false) == 1);  // a position no budget allows
    std::int64_t prev = model::LlamaGpu::kChunk;
    for (std::int64_t p0 = 1; p0 < (std::int64_t{1} << 40); p0 *= 4) {
      const std::int64_t t = g.chunk_tokens(p0, 64, true);
      CHECK(t >= 1);
      CHECK(t <= prev);
      prev = t;
    }
    CHECK(prev == 1);
  }

  // 08 doc step 6.6: with --trace, the GPU pass records one GPU-lane event
  // per layer and one for the LM head, from GPU timestamps mapped onto the
  // CPU clock (gpu/timestamps.h). They must sit inside the CPU-measured
  // window of the call, follow each other on the one queue, and change no
  // logit bit.
  TEST_CASE("gpu backend: --trace records every layer and the LM head on the GPU lane; logits unchanged") {
    if (!gpu::vulkan_available()) return;
    const gpu::ContextOptions opts = test::gpu_test_options();
    {
      const gpu::Context probe(opts);
      if (!probe.calibrated_timestamps_enabled()) {
        MESSAGE("gpu trace: skipped, the GPU lacks VK_EXT_calibrated_timestamps with the QPC domain");
        return;
      }
    }
    constexpr std::uint32_t kLayers = 3;
    const auto q = quantized_copy(test::tiny_llama(kLayers, 4, 2).random_f32_data(61).build(), kLayers, 4, 2, 8);
    const gguf::File f = gguf::File::parse(q);
    ThreadPool pool(2);
    std::vector<std::int32_t> tokens;
    for (int i = 0; i < 9; ++i) tokens.push_back((i * 5 + 1) % 32);
    const auto dir = test::repo_dir() / "build" / "test-tmp";
    std::filesystem::create_directories(dir);
    // The clock readings may be off by up to their deviation (Phase 1: up to
    // 160 us); 1 ms of slack still rejects a wrong period or offset.
    constexpr std::int64_t kSlackNs = 1'000'000;
    std::vector<float> plain, traced;
    for (const bool tracing : {false, true}) {
      CAPTURE(tracing);
      model::LlamaRef llama(f, 64, pool, model::Matmul::kGpu, Isa::kScalar, &opts);
      std::vector<float> logits(llama.config().n_vocab);
      if (tracing) trace::start();
      const std::int64_t t_start = now_ns();
      llama.forward(tokens, logits, false);
      const std::int64_t t_end = now_ns();
      (tracing ? traced : plain) = logits;
      if (!tracing) continue;
      std::vector<trace::Event> gpu;
      for (const trace::Event& e : trace::events()) {
        if (e.lane == trace::kGpuLane) gpu.push_back(e);
      }
      trace::write(dir / "gpu_pass_trace_test.json");  // also stops recording
      std::filesystem::remove(dir / "gpu_pass_trace_test.json");
      REQUIRE(gpu.size() == kLayers + 1);  // the layers in order, then the LM head
      for (std::size_t j = 0; j < gpu.size(); ++j) {
        CAPTURE(j);
        const trace::Event& e = gpu[j];
        const bool head = j == kLayers;
        CHECK(std::string_view(e.name) == (head ? "gpu lm_head" : "gpu layer"));
        CHECK(e.layer == (head ? -1 : static_cast<int>(j)));
        CHECK(e.tokens == (head ? 1 : static_cast<std::int64_t>(tokens.size())));  // the head: the last token only
        CHECK(e.t1_ns >= e.t0_ns);
        CHECK(e.t0_ns >= t_start - kSlackNs);
        CHECK(e.t1_ns <= t_end + kSlackNs);
        if (j > 0) CHECK(e.t0_ns >= gpu[j - 1].t1_ns - kSlackNs);  // one queue
      }
      CHECK(llama.gpu_pass()->validation_messages() == 0);
    }
    REQUIRE(plain.size() == traced.size());
    CHECK(std::memcmp(plain.data(), traced.data(), plain.size() * sizeof(float)) == 0);
  }

  // 08 doc section 4 (step 6.5): a lost GPU, or a submission over the
  // watchdog limit, switches a run with fallback to the cpu backend, which
  // re-runs the tokens so far: every later logit is the cpu backend's, bit
  // for bit. Without fallback the error ends the run. Only real events (here
  // the lowered watchdog) are written to the hang ledger, never the
  // simulated loss.
  TEST_CASE(
      "gpu backend: a lost GPU or a watchdog trip falls back to the cpu backend exactly; fails without "
      "fallback") {
    if (!gpu::vulkan_available()) return;
    const gpu::ContextOptions opts = test::gpu_test_options();
    const auto q = quantized_copy(test::tiny_llama(2, 4, 2).random_f32_data(61).build(), 2, 4, 2, 2);
    const gguf::File f = gguf::File::parse(q);
    const std::vector<std::int32_t> tokens = {1, 17, 3, 29, 5, 11, 7, 23, 13, 2};
    constexpr std::size_t kPrefill = 6;
    const auto cpu = run(q, 4, tokens, true, model::Matmul::kQ8Activations);
    const std::size_t V = cpu.size() / tokens.size();
    const auto dir = test::repo_dir() / "build" / "test-tmp";
    std::filesystem::create_directories(dir);
    const auto ledger = dir / "gpu_fallback_ledger.txt";
    std::filesystem::remove(ledger);
    struct LedgerAt {
      explicit LedgerAt(const std::filesystem::path& p) { gpu::set_hang_ledger_path(p); }
      ~LedgerAt() { gpu::set_hang_ledger_path({}); }
      LedgerAt(const LedgerAt&) = delete;
      LedgerAt& operator=(const LedgerAt&) = delete;
    } at(ledger);
    // A 6-token prefill, then 4 tokens one by one. Each call is 2 submissions
    // on this model (its layers, its LM head). `lose` < 0: the watchdog trips
    // on the first submission instead.
    for (const int lose : {0, 1, 2, 5, -1}) {
      CAPTURE(lose);
      ThreadPool pool(4);
      model::LlamaRef llama(f, 64, pool, model::Matmul::kGpu, Isa::kScalar, &opts);
      llama.set_gpu_fallback(true);
      if (lose >= 0) {
        llama.gpu_pass()->simulate_device_loss_after(static_cast<std::uint64_t>(lose));
      } else {
        llama.gpu_pass()->set_watchdog_ms(0.0);
      }
      std::vector<float> got(cpu.size());
      {
        // The fallback tells the user once, and a watchdog trip is also
        // written to the hang ledger, which says so; both are expected here.
        const LogCapture log;
        llama.forward(std::span(tokens).first(kPrefill), std::span(got).first(kPrefill * V), true);
        for (std::size_t i = kPrefill; i < tokens.size(); ++i) {
          llama.forward(std::span(tokens).subspan(i, 1), std::span(got).subspan(i * V, V), false);
        }
        CHECK(log.count("warning: gpu: the GPU failed (") == 1);
        CHECK(log.count("continuing on the CPU (--backend cpu)") == 1);
        if (lose >= 0) {
          CHECK(log.count("simulated device loss (test hook)") == 1);
          CHECK(log.lines() == 1);
        } else {
          CHECK(log.count("near-miss watchdog") == 1);
          CHECK(log.count("warning: GPU hang ledger: near-miss recorded in ") == 1);
          CHECK(log.lines() == 2);
        }
      }
      CHECK(llama.gpu_lost());
      CHECK(llama.gpu_pass() == nullptr);
      CHECK(llama.matmul() == model::Matmul::kQ8Activations);
      // The call that failed and every later one ran on the cpu backend. A
      // loss within the prefill (the first 2 submissions) makes all of it CPU.
      const std::size_t first_cpu = lose >= 2 ? kPrefill + static_cast<std::size_t>(lose - 2) / 2 : 0;
      CHECK(std::memcmp(cpu.data() + first_cpu * V, got.data() + first_cpu * V,
                        (tokens.size() - first_cpu) * V * sizeof(float)) == 0);
      CHECK(llama.n_past() == tokens.size());
    }
    // Only the watchdog trip was a real event.
    {
      std::ifstream in(ledger, std::ios::binary);
      std::vector<std::string> lines;
      for (std::string l; std::getline(in, l);) lines.push_back(l);
      REQUIRE(lines.size() == 1);
      CHECK(lines[0].find("\tnear-miss\t") != std::string::npos);
    }
    std::filesystem::remove(ledger);
    // Without fallback the loss is the run's error, and the GPU is not used again.
    ThreadPool pool(4);
    model::LlamaRef strict(f, 64, pool, model::Matmul::kGpu, Isa::kScalar, &opts);
    strict.gpu_pass()->simulate_device_loss_after(1);
    std::vector<float> logits(tokens.size() * V);
    bool lost = false;
    try {
      strict.forward(tokens, logits, true);
    } catch (const Error& e) {
      lost = e.kind() == ErrorKind::kDeviceLost;
    }
    CHECK(lost);
    CHECK_FALSE(strict.gpu_lost());
    CHECK(strict.gpu_pass() == nullptr);
    CHECK_FALSE(std::filesystem::exists(ledger));
    // The failed call left no trace, and the abandoned engine refuses to run
    // again (it has no CPU KV cache), even after reset().
    CHECK(strict.n_past() == 0);
    for (int attempt = 0; attempt < 2; ++attempt) {
      bool refused = false;
      try {
        strict.forward(tokens, logits, true);
      } catch (const Error& e) {
        refused = e.kind() == ErrorKind::kDeviceLost;
      }
      CHECK(refused);
      CHECK(strict.n_past() == 0);
      strict.reset();
    }
  }

  // Review finding m3: Ctrl+C stops a long prompt at a chunk boundary, the
  // context stays as before the call, and the press is consumed.
  TEST_CASE("cpu-ref: Ctrl+C during a long prompt stops at a chunk boundary and leaves the context unchanged") {
    const auto bytes = test::tiny_llama(1, 2, 1, 0, 64, 1024).random_f32_data(3).build();
    const gguf::File f = gguf::File::parse(bytes);
    ThreadPool pool(2);
    model::LlamaRef llama(f, 1024, pool);
    std::vector<std::int32_t> prompt(600);
    for (std::size_t i = 0; i < prompt.size(); ++i) prompt[i] = static_cast<std::int32_t>(1 + i % 30);
    std::vector<float> logits(32);
    const std::vector<std::int32_t> head(prompt.begin(), prompt.begin() + 10);
    llama.forward(head, logits, false);
    REQUIRE(llama.n_past() == 10);
    clear_cancel();
    request_cancel();
    CHECK_THROWS_WITH_AS(llama.forward(prompt, logits, false), doctest::Contains("Ctrl+C"), Error);
    CHECK(llama.n_past() == 10);  // the first 512-token chunk ran, and was rolled back
    // A prompt of one chunk is not interrupted mid-way (the decode loop polls instead).
    llama.forward(std::span(prompt).subspan(0, 5), logits, false);
    CHECK(llama.n_past() == 15);
    clear_cancel();
    // After the press is handled, the same long prompt runs.
    llama.forward(prompt, logits, false);
    CHECK(llama.n_past() == 615);
  }

  // Review finding m1: the gpu backend's attention buffers are checked
  // against the device limits before allocation, with the largest context
  // that fits. 1B shapes: 16 layers, K/V width 512, chunk 64, 32 heads.
  TEST_CASE("gpu backend: attention buffers above the device limits are a clear error naming the largest context") {
    const std::uint64_t GiB = 1ull << 30;
    // Iris Xe's limits (maxBufferSize 4 GiB, maxStorageBufferRange 1 GiB): ctx 32768 fits.
    CHECK_NOTHROW(model::check_attention_limits(16, 32768, 512, 64, 32, 4 * GiB, GiB));
    // ctx 131072: the K cache is 4 GiB, which is exactly maxBufferSize, so it fits; the score buffer, 1 GiB, fits too.
    CHECK_NOTHROW(model::check_attention_limits(16, 131072, 512, 64, 32, 4 * GiB, GiB));
    // One position more than the score buffer's range allows.
    CHECK_THROWS_WITH_AS(model::check_attention_limits(16, 131073, 512, 64, 32, 8 * GiB, GiB),
                         doctest::Contains("--ctx 131072 or less"), Error);
    // A smaller maxBufferSize limits the whole cache: 2 GiB / (16 x 512 x 4) = 65536.
    CHECK_THROWS_WITH_AS(model::check_attention_limits(16, 70000, 512, 64, 32, 2 * GiB, GiB),
                         doctest::Contains("--ctx 65536 or less"), Error);
    // An unknown maxBufferSize (0, Vulkan before 1.3) is not checked.
    CHECK_NOTHROW(model::check_attention_limits(16, 70000, 512, 64, 32, 0, GiB));
  }

  TEST_CASE("cpu-ref: context overflow and bad token ids are clear errors") {
    const auto bytes = test::tiny_llama().random_f32_data(1).build();
    const gguf::File f = gguf::File::parse(bytes);
    ThreadPool pool(1);
    model::LlamaRef llama(f, 4, pool);
    std::vector<float> logits(32);
    const std::vector<std::int32_t> five = {1, 2, 3, 4, 5};
    CHECK_THROWS_WITH_AS(llama.forward(five, logits, false), doctest::Contains("context is full"), Error);
    const std::vector<std::int32_t> bad = {99};
    CHECK_THROWS_WITH_AS(llama.forward(bad, logits, false), doctest::Contains("outside the vocabulary"), Error);
  }
}

TEST_SUITE("model") {
  TEST_CASE("cpu-ref and cpu: the real Q8_0 model completes 'The capital of France is' with ' Paris'") {
    const auto path = test::repo_dir() / "models" / "Llama-3.2-1B-Instruct-Q8_0-pure.gguf";
    if (!std::filesystem::exists(path)) {
      MESSAGE("missing (run scripts\\fetch-models.ps1): Llama-3.2-1B-Instruct-Q8_0-pure.gguf");
      return;
    }
    const gguf::File f = gguf::File::open(path);
    // Token ids from llama-tokenize (b11960) for "The capital of France is", BOS first.
    const std::vector<std::int32_t> tokens = {128000, 791, 6864, 315, 9822, 374};
    ThreadPool pool(4), one(1);
    for (const model::Matmul m : {model::Matmul::kFp32, model::Matmul::kQ8Activations}) {
      CAPTURE(static_cast<int>(m));
      model::LlamaRef llama(f, 64, pool, m);
      std::vector<float> logits(llama.config().n_vocab);
      llama.forward(tokens, logits, false);
      // ACCEPTANCE (Determinism): bit-identical at any thread count, on the real model too.
      model::LlamaRef single(f, 64, one, m);
      std::vector<float> again(logits.size());
      single.forward(tokens, again, false);
      CHECK(std::memcmp(logits.data(), again.data(), logits.size() * sizeof(float)) == 0);
      // ADR 0009 (prefill = decode): the same tokens fed one at a time give the same bits.
      model::LlamaRef stepwise(f, 64, pool, m);
      std::vector<float> step(logits.size());
      for (std::size_t i = 0; i < tokens.size(); ++i) stepwise.forward(std::span(tokens).subspan(i, 1), step, false);
      CHECK(std::memcmp(logits.data(), step.data(), logits.size() * sizeof(float)) == 0);
      const auto top = std::max_element(logits.begin(), logits.end()) - logits.begin();
      CHECK(top == 12366);  // " Paris"; the runner-up, " a", trails by 2.7 in llama.cpp
      // llama.cpp b11960 with an F32 KV cache (hetero-oracle-dump --f32-kv) gives 19.099213.
      // Measured on 2026-11-04: cpu-ref 19.1221, cpu 19.111992 (AVX-512 kernels),
      // 19.136883 (scalar kernels). With 8-bit activations, any FP32 rounding
      // difference (another compiler's libm, another summation order) moves
      // logits by up to ~0.4, so the bound
      // is loose; the layer-by-layer gates are scripts/oracle-check.ps1.
      CHECK(std::fabs(logits[12366] - 19.099213f) <= 0.25f);
    }
  }
}
