#include "model/speculative.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "core/clock.h"
#include "core/error.h"
#include "core/platform.h"
#include "core/trace.h"

namespace hearth::model {
namespace {

// One draw from a distribution whose probabilities sum to 1; the last
// candidate takes what rounding leaves.
std::int32_t draw(std::span<const sampling::Candidate> d, sampling::Sampler& s) {
  HEARTH_CHECK(!d.empty());
  const double u = s.uniform();
  double acc = 0;
  for (std::size_t i = 0; i + 1 < d.size(); ++i) {
    acc += d[i].p;
    if (u < acc) return d[i].id;
  }
  return d.back().id;
}

double probability_of(std::span<const sampling::Candidate> d, std::int32_t id) {
  for (const sampling::Candidate& c : d) {
    if (c.id == id) return c.p;
  }
  return 0.0;
}

// Observations above this multiple of the prediction are clipped: a pass that
// the operating system paused for a moment must not dominate the estimates.
constexpr double kOutlier = 3.0;
constexpr double kDecay = 0.95;          // weight of the earlier verification observations
constexpr double kDraftWeight = 0.1;     // of a new draft-step observation
constexpr double kPriorSlope = 0.06;     // extra token / target step while unfitted
constexpr double kPriorThreshold = 0.3;  // before any measurement

}  // namespace

void SpeculativeState::add_draft_step(double seconds) {
  if (!(seconds > 0)) return;
  if (draft_step_ <= 0) {
    draft_step_ = seconds;
    return;
  }
  draft_step_ += kDraftWeight * (std::min(seconds, kOutlier * draft_step_) - draft_step_);
}

void SpeculativeState::add_verification(int drafted, double seconds) {
  if (!(seconds > 0)) return;
  // A one-token pass runs other kernels (one row at a time) than a pass over
  // several tokens (4-row tiles), so it is measured on its own; the line is
  // fitted to the passes with drafts.
  if (drafted == 0) {
    one_step_ =
        one_step_ <= 0 ? seconds : one_step_ + kDraftWeight * (std::min(seconds, kOutlier * one_step_) - one_step_);
    return;
  }
  if (s0_ > 0) seconds = std::min(seconds, kOutlier * predicted(drafted));
  const double k = drafted;
  s0_ = kDecay * s0_ + 1;
  s1_ = kDecay * s1_ + k;
  s2_ = kDecay * s2_ + k * k;
  t0_ = kDecay * t0_ + seconds;
  t1_ = kDecay * t1_ + k * seconds;
}

double SpeculativeState::extra_token() const {
  if (s0_ <= 0) return kPriorSlope * one_step_;
  const double km = s1_ / s0_, tm = t0_ / s0_, var = s2_ / s0_ - km * km;
  // A fit needs draft lengths that vary; otherwise the prior slope.
  if (var < 0.25) return kPriorSlope * tm / (1 + kPriorSlope * km);
  const double slope = (t1_ / s0_ - km * tm) / var;
  return std::clamp(slope, 0.0, tm);
}

double SpeculativeState::intercept() const {
  if (s0_ <= 0) return one_step_;
  return std::max(t0_ / s0_ - extra_token() * s1_ / s0_, 1e-9);
}

double SpeculativeState::target_step() const { return one_step_ > 0 ? one_step_ : intercept(); }

double SpeculativeState::predicted(int drafted) const {
  return drafted == 0 ? target_step() : intercept() + drafted * extra_token();
}

double SpeculativeState::threshold() const {
  if (!measured()) return kPriorThreshold;
  return std::clamp((draft_step_ + extra_token()) / target_step(), 0.01, 1.0);
}

namespace detail {

double top_probability(std::span<const float> logits) {
  HEARTH_CHECK(!logits.empty());
  const float m = *std::max_element(logits.begin(), logits.end());
  // Logits more than 20 below the maximum add less than e^-20 each, under
  // 2.6e-4 of the sum for Llama's 128,256 tokens: negligible for a stopping
  // rule, and skipping their exp() saves most of the time (usually only a few
  // hundred logits are this close).
  const float floor = m - 20.0f;
  double sum = 0;
  for (const float l : logits) {
    if (l > floor) sum += static_cast<double>(std::exp(l - m));
  }
  return 1.0 / sum;
}

std::int32_t accept_or_resample(std::span<const sampling::Candidate> p, std::span<const sampling::Candidate> q,
                                std::int32_t d, sampling::Sampler& sampler, std::span<double> scratch, bool& accepted) {
  const double pd = probability_of(p, d), qd = probability_of(q, d);
  HEARTH_CHECK(qd > 0.0);  // d was drawn from q
  // u < p(d) / q(d), without the division.
  if (sampler.uniform() * qd < pd) {
    accepted = true;
    return d;
  }
  accepted = false;
  for (const sampling::Candidate& c : q) scratch[static_cast<std::size_t>(c.id)] = c.p;
  double total = 0;
  for (const sampling::Candidate& c : p) total += std::max(0.0, c.p - scratch[static_cast<std::size_t>(c.id)]);
  std::int32_t out = -1;
  if (total > 0.0) {
    const double u = sampler.uniform() * total;
    double acc = 0;
    for (const sampling::Candidate& c : p) {
      const double r = std::max(0.0, c.p - scratch[static_cast<std::size_t>(c.id)]);
      if (r <= 0.0) continue;
      out = c.id;  // the last token with a positive residual takes what rounding leaves
      acc += r;
      if (u < acc) break;
    }
  } else {
    // A rejection means p(d) < q(d), so some token has p > q and the residual
    // is positive in exact arithmetic; only rounding can leave it empty.
    out = draw(p, sampler);
  }
  for (const sampling::Candidate& c : q) scratch[static_cast<std::size_t>(c.id)] = 0.0;
  return out;
}

}  // namespace detail

GenerateResult generate_speculative(LlamaRef& target, LlamaRef& draft, const tokenizer::Tokenizer& tok,
                                    sampling::Sampler& sampler, std::span<const std::int32_t> prompt, int max_new,
                                    const SpeculativeOptions& options, const std::function<void(std::int32_t)>& emit,
                                    SpeculativeState* state) {
  GenerateResult r;
  r.prompt_tokens = static_cast<int>(prompt.size());
  if (prompt.empty()) fail(ErrorKind::kInvalidArgument, "generate: the prompt is empty");
  if (options.draft_max < 1 || options.draft_max > 16) {
    fail(ErrorKind::kInvalidArgument, "the draft length must be 1..16 (got {})", options.draft_max);
  }
  if (!(options.draft_p_min <= 1.0f)) {  // negative: automatic
    fail(ErrorKind::kInvalidArgument, "the draft's minimum probability must be at most 1 (got {})",
         options.draft_p_min);
  }
  if (draft.config().n_vocab != target.config().n_vocab) {
    fail(ErrorKind::kInvalidArgument, "the draft model has {} tokens and the target {}: they need one vocabulary",
         draft.config().n_vocab, target.config().n_vocab);
  }
  if (draft.n_ctx() < target.n_ctx()) {
    fail(ErrorKind::kInvalidArgument, "the draft's context ({}) is smaller than the target's ({})", draft.n_ctx(),
         target.n_ctx());
  }
  const std::uint32_t entry = target.n_past();
  SpeculativeState local;
  SpeculativeState& st = state != nullptr ? *state : local;
  if (draft.n_past() + st.skipped() + st.lag().size() != entry) {
    fail(ErrorKind::kInternal,
         "speculative: the draft holds {} positions, skipped {}, {} more are pending, and the target holds {}",
         draft.n_past(), st.skipped(), st.lag().size(), entry);
  }
  if (options.lookup_ngram < 0 || options.lookup_min < 1) {
    fail(ErrorKind::kInvalidArgument, "prompt lookup needs lookup_ngram >= 0 and lookup_min >= 1");
  }
  if (options.draft_read_max < 0 || options.draft_read_head < 0 ||
      (options.draft_read_max > 0 && options.draft_read_head >= options.draft_read_max)) {
    fail(ErrorKind::kInvalidArgument, "the draft's reading limit needs 0 <= head < max (or max 0)");
  }
  const std::uint32_t draft_entry = draft.n_past();
  const std::vector<std::int32_t> lag_on_entry = st.lag();
  const std::size_t skipped_on_entry = st.skipped();
  const std::uint32_t n_ctx = target.n_ctx();
  const std::uint32_t room = n_ctx - entry;
  if (room == 0) {  // as model::generate: a full context is a stop reason
    r.stop = Stop::kContextFull;
    return r;
  }
  if (prompt.size() > room) {
    fail(ErrorKind::kInvalidArgument, "the prompt needs {} positions but only {} of the {} are free", prompt.size(),
         room, n_ctx);
  }
  const std::size_t V = target.config().n_vocab;
  const auto kmax = static_cast<std::size_t>(options.draft_max);
  std::vector<float> logits((kmax + 1) * V), dlogits(V);
  const auto row = [&](std::size_t i) { return std::span<const float>(logits).subspan(i * V, V); };
  const std::int64_t t0 = now_ns();
  try {
    target.forward(prompt, std::span(logits).first(V), false);
  } catch (const Error& e) {
    // Ctrl+C during a long prompt: forward() left the context as it was.
    if (e.kind() != ErrorKind::kCancelled) throw;
    r.stop = Stop::kCancelled;
    clear_cancel();
    return r;
  }
  const std::int64_t t1 = now_ns();
  r.prefill_seconds = static_cast<double>(t1 - t0) * 1e-9;

  SpeculativeState& cost = st;
  const bool automatic = options.draft_p_min < 0.0f;
  try {
    const bool greedy = sampler.is_greedy();
    // The tokens the target holds after the draft's position on entry (the
    // lag, then this call's); the draft holds hist[0, dvalid) of them.
    std::vector<std::int32_t> hist = lag_on_entry;
    hist.insert(hist.end(), prompt.begin(), prompt.end());
    std::size_t dvalid = 0;
    std::size_t skip = 0;  // tokens of hist the draft skipped in this call: it holds hist[0, dvalid) less these
    std::vector<std::int32_t> feed, seq;
    std::vector<std::vector<sampling::Candidate>> q(kmax);  // the draft's distributions (sampling)
    std::vector<sampling::Candidate> p;
    std::vector<double> scratch(greedy ? 0 : V, 0.0);
    std::int32_t id = -1;  // the target's next token, not yet emitted
    // The loop of model::generate, one target pass per iteration: `id`, then
    // the kept draft tokens, each through the same checks in the same order.
    while (r.generated < max_new) {
      if (cancel_requested()) {
        r.stop = Stop::kCancelled;
        clear_cancel();  // consumed by the call it stopped: the next call runs
        break;
      }
      if (id < 0) id = sampler.sample(row(0));
      if (tok.is_eog(id)) {
        r.stop = Stop::kEndOfGeneration;
        r.eog_token = id;
        break;
      }
      emit(id);
      ++r.generated;
      const std::uint32_t base = target.n_past();
      if (base >= n_ctx) {
        r.stop = Stop::kContextFull;
        break;
      }
      if (r.generated >= max_new) {  // model::generate feeds the last token too
        const std::int64_t l0 = now_ns();
        target.forward(std::span(&id, 1), std::span(logits).first(V), false);
        cost.add_verification(0, static_cast<double>(now_ns() - l0) * 1e-9);
        hist.push_back(id);
        r.stop = Stop::kLimit;
        break;
      }
      // Draft up to k tokens after id: no more than may still be emitted, and
      // no more than the context holds after id.
      // Draft up to k tokens after id: no more than may still be emitted, and
      // no more than the context holds after id.
      const int k = std::min({options.draft_max, max_new - r.generated, static_cast<int>(n_ctx - base - 1)});
      seq.assign(1, id);
      bool looked_up = false;
      if (k > 0 && options.lookup_ngram > 0) {
        // Prompt lookup: the latest earlier occurrence of the last n tokens
        // of the text (hist, then id), and up to k of the tokens after it.
        const auto n = static_cast<std::size_t>(options.lookup_ngram);
        const std::size_t total = hist.size() + 1;
        const auto at = [&](std::size_t i) { return i < hist.size() ? hist[i] : id; };
        // `end`: where the earlier occurrence ends, from total - 2 down to n - 1.
        for (std::size_t end = total - 1; total > n && end-- > n - 1;) {
          bool match = true;
          for (std::size_t m = 0; m < n && match; ++m) match = at(end - m) == at(total - 1 - m);
          if (!match) continue;
          for (std::size_t c = end + 1; c < total && seq.size() < static_cast<std::size_t>(k) + 1; ++c) {
            seq.push_back(at(c));
          }
          break;
        }
        if (seq.size() - 1 >= static_cast<std::size_t>(options.lookup_min)) {
          looked_up = true;
          if (!greedy) {  // a proposal of probability 1 for the rejection rule
            for (std::size_t i = 0; i + 1 < seq.size(); ++i) q[i].assign(1, {seq[i + 1], 1.0});
          }
        } else {
          seq.resize(1);
        }
      }
      if (k > 0 && !looked_up) {
        const trace::Scope ts("draft", -1, k, 0);
        // The draft catches up with what the target holds, then reads id.
        feed.assign(hist.begin() + static_cast<std::ptrdiff_t>(dvalid), hist.end());
        feed.push_back(id);
        const auto read_max = static_cast<std::size_t>(options.draft_read_max);
        if (state != nullptr && read_max > 0 && feed.size() > read_max) {
          // A long prompt: its first draft_read_head tokens and the most recent rest.
          const auto head = static_cast<std::size_t>(options.draft_read_head);
          const std::size_t cut = feed.size() - read_max;
          feed.erase(feed.begin() + static_cast<std::ptrdiff_t>(head),
                     feed.begin() + static_cast<std::ptrdiff_t>(head + cut));
          skip += cut;
        }
        std::int64_t s0 = now_ns();
        draft.forward(feed, dlogits, false);
        if (feed.size() == 1) cost.add_draft_step(static_cast<double>(now_ns() - s0) * 1e-9);
        const double threshold = cost.threshold();
        double all_right = 1.0;  // the draft's probability that its tokens so far are all right
        for (int i = 0; i < k; ++i) {
          if (automatic || options.draft_p_min > 0.0f) {
            const double top = detail::top_probability(dlogits);
            all_right *= top;
            if (automatic ? all_right < threshold : top < options.draft_p_min) break;
          }
          std::int32_t d = 0;
          if (greedy) {
            d = sampling::greedy(dlogits);
          } else {
            sampler.distribution(dlogits, q[static_cast<std::size_t>(i)]);
            d = draw(q[static_cast<std::size_t>(i)], sampler);
          }
          seq.push_back(d);
          if (i + 1 < k) {
            s0 = now_ns();
            draft.forward(std::span(&seq.back(), 1), dlogits, false);
            cost.add_draft_step(static_cast<double>(now_ns() - s0) * 1e-9);
          }
        }
      }
      // One target pass over id and the drafts: row i holds the target's
      // logits after seq[0..i], bit for bit those of one-token steps.
      const int kd = static_cast<int>(seq.size()) - 1;
      const std::int64_t v0 = now_ns();
      target.forward(seq, std::span(logits).first(seq.size() * V), true);
      cost.add_verification(kd, static_cast<double>(now_ns() - v0) * 1e-9);
      r.drafted += kd;
      if (looked_up) r.looked_up += kd;
      int keep = 0;  // draft tokens that stay in the context
      bool stopped = false;
      std::int32_t next = -1;
      for (int i = 0; i < kd; ++i) {
        const std::int32_t d = seq[static_cast<std::size_t>(i) + 1];
        bool ok = false;
        if (greedy) {
          next = sampling::greedy(row(static_cast<std::size_t>(i)));
          ok = next == d;
        } else {
          sampler.distribution(row(static_cast<std::size_t>(i)), p);
          next = detail::accept_or_resample(p, q[static_cast<std::size_t>(i)], d, sampler, scratch, ok);
        }
        if (!ok) break;  // `next` replaces d
        ++r.accepted;
        // d is the target's token: model::generate's checks for it. It is fed
        // already; a token that ends the reply or is not reached is dropped.
        if (cancel_requested()) {
          r.stop = Stop::kCancelled;
          clear_cancel();
          stopped = true;
          break;
        }
        if (tok.is_eog(d)) {
          r.stop = Stop::kEndOfGeneration;
          r.eog_token = d;
          stopped = true;
          break;
        }
        emit(d);
        ++r.generated;
        keep = i + 1;
        if (r.generated >= max_new) {
          r.stop = Stop::kLimit;
          stopped = true;
          break;
        }
      }
      if (!stopped && keep == kd) {  // every draft kept: the pass also gives the token after them
        next = greedy ? sampling::greedy(row(static_cast<std::size_t>(kd)))
                      : sampler.sample(row(static_cast<std::size_t>(kd)));
      }
      // The target keeps id and the kept drafts; so does the draft, as far as
      // it read them (it reads the last draft token only in the next cycle).
      target.truncate(base + 1 + static_cast<std::uint32_t>(keep));
      const std::size_t before = hist.size();
      hist.push_back(id);
      hist.insert(hist.end(), seq.begin() + 1, seq.begin() + 1 + keep);
      if (k > 0 && !looked_up) {                // after a lookup the draft read nothing this cycle
        const int fed = kd == k ? kd - 1 : kd;  // drafts the draft read
        dvalid = before + 1 + static_cast<std::size_t>(std::min(fed, keep));
        draft.truncate(draft_entry + static_cast<std::uint32_t>(dvalid - skip));
      }
      if (stopped) break;
      id = next;
      r.stop = Stop::kLimit;
    }
    // With a kept state the draft reads the rest when it next drafts;
    // otherwise it ends holding what the target holds.
    if (state != nullptr) {
      st.lag().assign(hist.begin() + static_cast<std::ptrdiff_t>(dvalid), hist.end());
      st.set_skipped(skipped_on_entry + skip);
    } else if (dvalid < hist.size()) {
      feed.assign(hist.begin() + static_cast<std::ptrdiff_t>(dvalid), hist.end());
      draft.forward(feed, dlogits, false);
    }
    HEARTH_CHECK(target.n_past() == draft_entry + skipped_on_entry + hist.size());
    HEARTH_CHECK(draft.n_past() + (state != nullptr ? st.skipped() + st.lag().size() : 0) == target.n_past());
  } catch (...) {
    target.truncate(entry);
    draft.truncate(std::min(draft.n_past(), draft_entry));
    st.lag() = lag_on_entry;
    st.set_skipped(skipped_on_entry);
    throw;
  }
  r.decode_seconds = static_cast<double>(now_ns() - t1) * 1e-9;
  return r;
}

}  // namespace hearth::model
