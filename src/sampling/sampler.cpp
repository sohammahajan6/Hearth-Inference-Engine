#include "sampling/sampler.h"

#include <algorithm>
#include <cmath>

#include "core/error.h"

namespace hearth::sampling {
namespace {

std::uint64_t splitmix64(std::uint64_t& x) noexcept {
  std::uint64_t z = (x += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

std::uint64_t rotl(std::uint64_t x, int k) noexcept { return (x << k) | (x >> (64 - k)); }

}  // namespace

Rng::Rng(std::uint64_t seed) {
  for (std::uint64_t& s : s_) s = splitmix64(seed);
}

std::uint64_t Rng::next() noexcept {
  const std::uint64_t result = rotl(s_[1] * 5, 7) * 9;
  const std::uint64_t t = s_[1] << 17;
  s_[2] ^= s_[0];
  s_[3] ^= s_[1];
  s_[1] ^= s_[2];
  s_[0] ^= s_[3];
  s_[2] ^= t;
  s_[3] = rotl(s_[3], 45);
  return result;
}

double Rng::uniform() noexcept { return static_cast<double>(next() >> 11) * 0x1.0p-53; }

std::int32_t greedy(std::span<const float> logits) {
  HEARTH_CHECK(!logits.empty());
  std::size_t best = 0;
  for (std::size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > logits[best]) best = i;
  }
  return static_cast<std::int32_t>(best);
}

Sampler::Sampler(const Params& p) : p_(p), rng_(p.seed) {
  if (!(p.temperature >= 0.0f) || !std::isfinite(p.temperature)) {
    fail(ErrorKind::kInvalidArgument, "temperature must be a finite number of at least 0 (got {})", p.temperature);
  }
  if (!(p.top_p > 0.0f)) fail(ErrorKind::kInvalidArgument, "top-p must be above 0 (got {})", p.top_p);
}

std::size_t Sampler::select(std::span<const float> logits) {
  cand_.clear();
  cand_.reserve(logits.size());
  for (std::size_t i = 0; i < logits.size(); ++i) {
    // A NaN would break the sort's strict weak order (undefined behaviour).
    if (std::isnan(logits[i])) fail(ErrorKind::kInternal, "sampling: the logit of token {} is NaN", i);
    cand_.emplace_back(logits[i], static_cast<std::int32_t>(i));
  }
  // Descending logit, then ascending ID: a total order, so the result does
  // not depend on the sort implementation.
  const auto before = [](const std::pair<float, std::int32_t>& a, const std::pair<float, std::int32_t>& b) {
    return a.first != b.first ? a.first > b.first : a.second < b.second;
  };
  std::size_t n = cand_.size();
  if (p_.top_k > 0 && static_cast<std::size_t>(p_.top_k) < n) {
    n = static_cast<std::size_t>(p_.top_k);
    std::partial_sort(cand_.begin(), cand_.begin() + static_cast<std::ptrdiff_t>(n), cand_.end(), before);
  } else {
    std::sort(cand_.begin(), cand_.end(), before);
  }
  if (p_.top_p < 1.0f) {
    // Probabilities at temperature 1; keep the smallest prefix reaching top_p.
    double sum = 0;
    for (std::size_t i = 0; i < n; ++i) sum += std::exp(static_cast<double>(cand_[i].first - cand_[0].first));
    double acc = 0;
    std::size_t keep = n;
    for (std::size_t i = 0; i < n; ++i) {
      acc += std::exp(static_cast<double>(cand_[i].first - cand_[0].first)) / sum;
      if (acc >= p_.top_p) {
        keep = i + 1;
        break;
      }
    }
    n = keep;
  }
  return n;
}

void Sampler::distribution(std::span<const float> logits, std::vector<Candidate>& out) {
  out.clear();
  if (is_greedy()) {
    out.push_back({sampling::greedy(logits), 1.0});
    return;
  }
  const std::size_t n = select(logits);
  // The weights of sample(), normalized.
  double total = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const double w = std::exp(static_cast<double>(cand_[i].first - cand_[0].first) / p_.temperature);
    out.push_back({cand_[i].second, w});
    total += w;
  }
  for (Candidate& c : out) c.p /= total;
}

std::int32_t Sampler::sample(std::span<const float> logits) {
  if (p_.temperature <= 0.0f) return greedy(logits);
  const std::size_t n = select(logits);
  // One draw from softmax(logit / temperature) over the n candidates.
  std::vector<double> w(n);
  double total = 0;
  for (std::size_t i = 0; i < n; ++i) {
    w[i] = std::exp(static_cast<double>(cand_[i].first - cand_[0].first) / p_.temperature);
    total += w[i];
  }
  // Walk the cumulative weights; the last candidate takes what is left.
  const double u = rng_.uniform() * total;
  double acc = 0;
  for (std::size_t i = 0; i + 1 < n; ++i) {
    acc += w[i];
    if (u < acc) return cand_[i].second;
  }
  return cand_[n - 1].second;
}

}  // namespace hearth::sampling
