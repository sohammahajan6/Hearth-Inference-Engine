// Choosing the next token from the logits
//.
#pragma once

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace hearth::sampling {

struct Params {
  float temperature = 0.8f;  // <= 0: greedy
  int top_k = 40;            // <= 0: no top-k
  float top_p = 0.95f;       // >= 1: no top-p
  std::uint64_t seed = 0;
};

// xoshiro256** (Blackman and Vigna), seeded through splitmix64.
class Rng {
 public:
  explicit Rng(std::uint64_t seed);
  std::uint64_t next() noexcept;
  double uniform() noexcept;  // [0, 1), 53 random bits

 private:
  std::uint64_t s_[4];
};

// One token of a sampling distribution and its probability.
struct Candidate {
  std::int32_t id;
  double p;
};

class Sampler {
 public:
  explicit Sampler(const Params& p);
  // Greedy (temperature <= 0): the arg-max, the lowest ID on a tie.
  // Otherwise: top-k, then top-p on the probabilities at temperature 1, then
  // one draw from softmax(logit / temperature) over what is left (the order
  // of llama.cpp's default chain).
  std::int32_t sample(std::span<const float> logits);
  // The distribution sample() draws from: the candidates left after top-k and
  // top-p, with their probabilities at the temperature (summing to 1), most
  // likely first. Greedy: the arg-max alone, with probability 1. Speculative
  // decoding (model/speculative.h) compares the target's and the draft's.
  void distribution(std::span<const float> logits, std::vector<Candidate>& out);
  // One draw from [0, 1), from the same random sequence as sample().
  double uniform() noexcept { return rng_.uniform(); }
  bool is_greedy() const noexcept { return p_.temperature <= 0.0f; }

 private:
  // Fills cand_ and returns how many of its first entries survive top-k and
  // top-p, sorted by descending logit, then ascending ID.
  std::size_t select(std::span<const float> logits);

  Params p_;
  Rng rng_;
  std::vector<std::pair<float, std::int32_t>> cand_;
};

// The arg-max of `logits`, the lowest index on a tie.
std::int32_t greedy(std::span<const float> logits);

}  // namespace hearth::sampling
