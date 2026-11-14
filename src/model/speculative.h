// Speculative decoding: a small draft
// model proposes the next few tokens one at a time, and the target model
// checks all of them in one forward pass. Decode reads every weight of the
// target once per pass, so a pass over 1 + k tokens costs little more than a
// pass over one, and every accepted draft token saves one pass.
//
// The target decides every token:
// - Greedy (temperature 0): a draft token is kept only if it is the target's
//   arg-max. A multi-token pass gives exactly the logits of one-token steps
//  , so the emitted tokens, the
//   stop reason and the context are exactly those of model::generate.
// - Sampling: the rejection rule of Leviathan et al. and Chen et al. (2023).
//   A draft token d drawn from the draft's distribution q is kept with
//   probability min(1, p(d) / q(d)), where p is the target's; at the first
//   rejection the token is drawn from max(0, p - q), normalized. Each emitted
//   token then follows the target's distribution p exactly, as in
//   model::generate. p and q are the sampler's distributions after top-k,
//   top-p and the temperature. The random draws differ from model::generate's,
//   so one seed gives different (equally distributed) text.
#pragma once

#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include "model/generate.h"
#include "model/llama_ref.h"
#include "sampling/sampler.h"
#include "tokenizer/tokenizer.h"

namespace hearth::model {

struct SpeculativeOptions {
  // Draft tokens per target pass, at most (1..16).
  int draft_max = 8;
  // When to stop drafting before draft_max. Negative (the default): by
  // measured cost (SpeculativeState): draft the next token while the draft's
  // probability that all its tokens so far are right is at least
  // (draft step + one more verified token) / target step, so a token is
  // drafted only when its expected saving exceeds its cost. 0: never stop
  // early. Otherwise: stop when the draft's most likely next token has a
  // probability below this. Probabilities are softmax(draft logits) at
  // temperature 1.
  float draft_p_min = -1.0f;
  // With a kept state, the longest stretch of new tokens the draft reads in
  // one catch-up (a long prompt): beyond it, the first draft_read_head of
  // them and the most recent rest; the middle is skipped. The draft then
  // guesses from less context, which only changes how often it is right,
  // and a 1,000-token prompt costs it a 512-token pass instead of 1,000.
  // 0: read everything.
  int draft_read_max = 512;
  int draft_read_head = 64;
  // Prompt lookup: when the last lookup_ngram tokens (the next token
  // included) occurred earlier in this call's text (the prompt and the reply
  // so far), the tokens that followed them there are proposed instead of the
  // draft model's, at no draft cost, if there are at least lookup_min of
  // them. The draft model reads those tokens later. 0: off (the default:
  // on the 3B Q4_0, whose extra verified tokens are expensive, lookup made
  // decoding slower, 0.983x against 1.031x without; 13-speculative.md).
  int lookup_ngram = 0;
  int lookup_min = 2;
};

// What generate_speculative keeps between calls (an Engine keeps one): the
// costs that the automatic draft length weighs, measured as decoding runs on
// this machine, and the tokens the target holds that the draft has not read
// yet (it reads them when it next drafts, so the end of a reply costs the
// draft nothing: a one-shot reply never needs them).
class SpeculativeState {
 public:
  // The target's tokens after the draft's last position, in order.
  const std::vector<std::int32_t>& lag() const noexcept { return lag_; }
  std::vector<std::int32_t>& lag() noexcept { return lag_; }
  // Tokens of the target's context the draft skipped (draft_read_max):
  // draft positions + skipped() + lag().size() = target positions.
  std::size_t skipped() const noexcept { return skipped_; }
  void set_skipped(std::size_t n) noexcept { skipped_ = n; }
  // When both contexts are emptied (Engine::reset): the costs stay.
  void forget_context() noexcept {
    lag_.clear();
    skipped_ = 0;
  }

  void add_draft_step(double seconds);
  void add_verification(int drafted, double seconds);
  bool measured() const noexcept { return draft_step_ > 0 && (s0_ > 0 || one_step_ > 0); }
  double draft_step() const noexcept { return draft_step_; }
  // A one-token target pass: a running average of the one-token passes, or
  // before there is one, the fit's intercept. The cost of each further token
  // in a pass: the slope of a decayed least-squares line through the times of
  // the passes with drafts (a prior slope of 6 % per token while their
  // lengths do not vary enough).
  double target_step() const;
  double extra_token() const;
  // (draft step + extra token) / target step; 0.3 before any measurement.
  double threshold() const;

 private:
  double predicted(int drafted) const;
  double intercept() const;  // of the line through the passes with drafts
  double draft_step_ = 0;
  double one_step_ = 0;                                // running average of one-token passes
  double s0_ = 0, s1_ = 0, s2_ = 0, t0_ = 0, t1_ = 0;  // decayed sums of 1, k, k^2, t, k t
  std::vector<std::int32_t> lag_;
  std::size_t skipped_ = 0;
};

// model::generate with a draft model of the same vocabulary. The draft holds
// the target's tokens up to `state->lag()`, which holds the rest (without a
// state: exactly the target's tokens, on entry and on return). The result's
// `drafted` and `accepted` count the draft tokens proposed and kept. A call
// that throws leaves both contexts, and the lag, as they were before it.
// `state` (nullptr: one for this call only, and the draft catches up before
// returning) is updated here.
GenerateResult generate_speculative(LlamaRef& target, LlamaRef& draft, const tokenizer::Tokenizer& tok,
                                    sampling::Sampler& sampler, std::span<const std::int32_t> prompt, int max_new,
                                    const SpeculativeOptions& options, const std::function<void(std::int32_t)>& emit,
                                    SpeculativeState* state = nullptr);

namespace detail {
// The rejection step for one draft token `d` drawn from q (sampling only):
// keeps d with probability min(1, p(d) / q(d)) and returns it with
// `accepted` = true; otherwise returns a draw from max(0, p - q) normalized,
// with `accepted` = false. p and q list each token at most once. `scratch`
// holds n_vocab zeros on entry and on return. Public for the tests.
std::int32_t accept_or_resample(std::span<const sampling::Candidate> p, std::span<const sampling::Candidate> q,
                                std::int32_t d, sampling::Sampler& sampler, std::span<double> scratch, bool& accepted);
// The probability of the most likely token of softmax(logits), ignoring
// logits more than 20 below the maximum (relative error below 2.6e-4 for
// 128,256 tokens).
double top_probability(std::span<const float> logits);
}  // namespace detail

}  // namespace hearth::model
