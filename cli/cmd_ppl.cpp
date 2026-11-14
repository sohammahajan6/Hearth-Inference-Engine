// `hearth ppl`: perplexity with llama-perplexity's protocol (llama.cpp b11960,
// tools/perplexity/perplexity.cpp, ppl_stride 0), so the two can be compared
// number for number:
//  - the file is read, one trailing '\n' is dropped (llama.cpp's -f does that;
//    it also reads in text mode, which differs only for files with CR LF),
//    and the whole text is tokenized once, with BOS;
//  - chunk i is tokens [i*ctx, (i+1)*ctx); its first token is replaced by BOS
//    for the evaluation; each chunk starts from an empty KV cache;
//  - the second half is scored: the logits at positions ctx/2 .. ctx-2
//    predict the (original) next token;
//  - nll = max + log(sum exp(logit - max)) - logit[target], the sum in double
//    of float exponentials, as llama.cpp computes it.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "commands.h"
#include "core/clock.h"
#include "core/error.h"
#include "core/log.h"
#include "core/platform.h"
#include "session.h"

namespace hearth::cli {
namespace {

// -log p(target) with the arithmetic of llama.cpp's log_softmax().
double nll(const float* logits, std::int32_t n_vocab, std::int32_t target) {
  float max = logits[0];
  for (std::int32_t i = 1; i < n_vocab; ++i) max = std::max(max, logits[i]);
  double sum = 0.0;
  for (std::int32_t i = 0; i < n_vocab; ++i) sum += std::exp(logits[i] - max);
  return -(static_cast<double>(logits[target] - max) - std::log(sum));
}

}  // namespace

int cmd_ppl(Args& args) {
  const ModelOptions mo = model_options(args, 512);
  const auto file = args.value("file");
  const int chunks = static_cast<int>(args.int_or("chunks", -1, -1, 1 << 20));
  args.finish();
  if (!file) fail(ErrorKind::kInvalidArgument, "ppl needs --file F (for example data\\wikitext-2-raw\\wiki.test.raw)");
  std::ifstream in(path_from_utf8(*file), std::ios::binary);
  if (!in) fail(ErrorKind::kIo, "cannot read '{}'", *file);
  std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (!text.empty() && text.back() == '\n') text.pop_back();

  Session s(mo);
  const std::int32_t n_ctx = static_cast<std::int32_t>(mo.ctx);
  const std::vector<std::int32_t> tokens = s.tok.encode(text, s.tok.add_bos());
  if (tokens.size() < 2 * static_cast<std::size_t>(n_ctx)) {
    fail(ErrorKind::kInvalidArgument, "the file has {} tokens; perplexity with --ctx {} needs at least {}",
         tokens.size(), n_ctx, 2 * n_ctx);
  }
  const int n_chunk_max = static_cast<int>(tokens.size() / static_cast<std::size_t>(n_ctx));
  const int n_chunk = chunks < 0 ? n_chunk_max : std::min(chunks, n_chunk_max);
  const std::int32_t n_vocab = static_cast<std::int32_t>(s.llm.config().n_vocab);
  const std::int32_t first = n_ctx / 2;
  log_info("ppl: {} tokens, {} chunks of {}, {}, {} threads", tokens.size(), n_chunk, n_ctx, mo.backend, mo.threads);

  std::vector<float> last(static_cast<std::size_t>(n_vocab));
  std::vector<float> logits(static_cast<std::size_t>(n_ctx - first) * static_cast<std::size_t>(n_vocab));
  double sum = 0.0, sum2 = 0.0;
  long long count = 0;
  const std::int64_t t0 = now_ns();
  for (int i = 0; i < n_chunk; ++i) {
    throw_if_cancelled();
    const std::size_t start = static_cast<std::size_t>(i) * static_cast<std::size_t>(n_ctx);
    std::vector<std::int32_t> chunk(
        tokens.begin() + static_cast<std::ptrdiff_t>(start),
        tokens.begin() + static_cast<std::ptrdiff_t>(start + static_cast<std::size_t>(n_ctx)));
    if (s.tok.add_bos()) chunk[0] = s.tok.bos();
    s.llm.reset();
    s.llm.forward(std::span(chunk).first(static_cast<std::size_t>(first)), last, false);
    s.llm.forward(std::span(chunk).subspan(static_cast<std::size_t>(first)), logits, true);
    for (std::int32_t k = 0; k < n_ctx - first - 1; ++k) {
      const double v = nll(logits.data() + static_cast<std::size_t>(k) * static_cast<std::size_t>(n_vocab), n_vocab,
                           tokens[start + static_cast<std::size_t>(first + k + 1)]);
      sum += v;
      sum2 += v * v;
    }
    count += n_ctx - first - 1;
    if (i == 0) {
      log_info("ppl: {:.2f} seconds per chunk", static_cast<double>(now_ns() - t0) * 1e-9);
    }
    std::printf("[%d]%.4f,", i + 1, std::exp(sum / static_cast<double>(count)));
    std::fflush(stdout);
  }
  const double mean = sum / static_cast<double>(count);
  const double ppl = std::exp(mean);
  const double var = sum2 / static_cast<double>(count) - mean * mean;
  std::printf("\nFinal estimate: PPL = %.4f +/- %.5f\n", ppl,
              var > 0 ? std::sqrt(var / static_cast<double>(count - 1)) * ppl : 0.0);
  log_info("ppl: {} chunks in {:.1f} s", n_chunk, static_cast<double>(now_ns() - t0) * 1e-9);
  return 0;
}

}  // namespace hearth::cli
