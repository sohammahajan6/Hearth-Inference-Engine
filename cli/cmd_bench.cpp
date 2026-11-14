// `hearth bench`: prefill and decode speed, in llama-bench's terms.
//   ppN: one forward pass over N prompt tokens (BOS, then pseudo-random IDs),
//        logits for the last token only; tok/s = N / time.
//   tgN: N forward passes of one token each, from an empty context; tok/s = N / time.
// Every repetition starts from an empty KV cache. One unmeasured warm-up
// precedes each test; the median and the interquartile range (25th..75th
// percentile) of the repetitions are reported.
// With --bandwidth GBPS (the read bandwidth `hearth probe` measured), the
// ACCEPTANCE sanity rule is applied: decode tok/s x bytes per token must not
// exceed 1.05 x that bandwidth; a higher number means the measurement is wrong.
// With --prompts FILE and --draft D (speculative decoding, docs/architecture/
// 13-speculative.md): decode on real text. Each line of FILE is one chat
// prompt (user turn, fixed date). For every prompt and repetition, greedy
// decoding of up to --gen tokens runs once without and once with the draft,
// in alternating order (A B, then B A), in one process; decode tok/s =
// generated tokens / decode time, where the speculative decode time includes
// the draft reading the prompt. The two must give identical tokens, or the
// command fails.
// With --counters, performance counters are sampled every 100 ms during the
// measured repetitions of each test and their medians
// printed: the CPU clock relative to base, this process's GPU engine
// utilisation and the package power (RAPL). They show why a configuration is
// fast or slow, for example the package power limit (RISKS R12). Off by default: the sampling thread costs a little CPU
// time, so runs with and without it are compared only among themselves.
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <format>
#include <fstream>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "commands.h"
#include "core/environment.h"
#include "core/error.h"
#include "core/pdh.h"
#include "core/platform.h"
#include "hearth/version.h"
#include "model/generate.h"
#include "model/llama.h"
#include "model/speculative.h"
#include "session.h"

namespace hearth::cli {
namespace {

// Percentile of sorted values, by linear interpolation between ranks.
double percentile(const std::vector<double>& sorted, double p) {
  const double pos = p * static_cast<double>(sorted.size() - 1);
  const auto lo = static_cast<std::size_t>(pos);
  const std::size_t hi = std::min(lo + 1, sorted.size() - 1);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - static_cast<double>(lo));
}

struct Stats {
  double median, p25, p75;
};

Stats stats(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return {percentile(v, 0.5), percentile(v, 0.25), percentile(v, 0.75)};
}

// Samples the counters on a thread of its own between start() and stop().
class Counters {
 public:
  void start() {
    samples_.clear();
    stop_ = false;
    thread_ = std::thread([this] { run(); });
  }
  // The medians since start(); NaN for a counter this machine lacks.
  struct Medians {
    double cpu_perf = NAN, gpu_busy = NAN, package_w = NAN;
    std::size_t samples = 0;
  };
  Medians stop() {
    stop_ = true;
    thread_.join();
    Medians m;
    m.samples = samples_.size();
    auto median = [&](double Sample::* field) {
      std::vector<double> v;
      for (const Sample& s : samples_) {
        if (!std::isnan(s.*field)) v.push_back(s.*field);
      }
      return v.empty() ? NAN : stats(v).median;
    };
    m.cpu_perf = median(&Sample::cpu_perf);
    m.gpu_busy = median(&Sample::gpu_busy);
    m.package_w = median(&Sample::package_w);
    return m;
  }

 private:
  struct Sample {
    double cpu_perf, gpu_busy, package_w;
  };
  void run() {
    PdhQuery pdh;
    const int perf = pdh.add(L"\\Processor Information(_Total)\\% Processor Performance");
    const int gpu = pdh.add(L"\\GPU Engine(*)\\Utilization Percentage");
    const int pkg = pdh.add(L"\\Energy Meter(rapl_package0_pkg)\\Power");  // mW
    const std::string pid = std::format("pid_{}_", GetCurrentProcessId());
    pdh.collect();  // rate counters need a first sample
    while (!stop_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      pdh.collect();
      samples_.push_back({perf >= 0 ? pdh.sum(perf, {}) : NAN, gpu >= 0 ? pdh.sum(gpu, {pid}) : NAN,
                          pkg >= 0 ? pdh.sum(pkg, {}) / 1000.0 : NAN});
    }
  }
  std::thread thread_;
  std::atomic<bool> stop_{false};
  std::vector<Sample> samples_;
};

std::string format_counters(const Counters::Medians& m) {
  auto field = [](double v, const char* unit) {
    return std::isnan(v) ? std::string("not measured") : std::format("{:.0f}{}", v, unit);
  };
  return std::format("CPU clock {} of base, GPU busy {} (this process, all engines), package {}; {} samples",
                     field(m.cpu_perf, " %"), field(m.gpu_busy, " %"), field(m.package_w, " W"), m.samples);
}

// Release build and AC power, unless allowed otherwise.
void check_conditions(bool allow_battery, bool allow_non_release) {
  if (build_info().build_type != std::string("Release") && !allow_non_release) {
    fail(ErrorKind::kInvalidArgument,
         "bench measures performance; use a Release build (this is {}; --allow-non-release runs it anyway)",
         build_info().build_type);
  }
  if (power_source() != PowerSource::kAc && !allow_battery) {
    fail(ErrorKind::kInvalidArgument,
         "the machine is not on AC power; battery limits clocks and makes numbers incomparable "
         "(pass --allow-battery to measure anyway)");
  }
}

int bench_speculative(Args& args, const std::string& prompts_path) {
  const int gen = static_cast<int>(args.int_or("gen", 128, 1, 1 << 16));
  const int reps = static_cast<int>(args.int_or("reps", 1, 1, 100));
  const int round_offset = static_cast<int>(args.int_or("round-offset", 0, 0, 1));
  // Greedy by default; with --temp > 0 both runs of a pair sample with the
  // same options and seed (their texts then differ, with equal distributions).
  const auto temp = static_cast<float>(args.double_or("temp", 0.0, 0.0, 100.0));
  const int top_k = static_cast<int>(args.int_or("top-k", 40, 0, 1 << 20));
  const auto top_p = static_cast<float>(args.double_or("top-p", 0.95, 1e-6, 1.0));
  const auto seed = static_cast<std::uint64_t>(args.int_or("seed", 1, 0, 1LL << 62));
  const bool allow_battery = args.flag("allow-battery");
  const bool allow_non_release = args.flag("allow-non-release");
  ModelOptions mo = model_options(args, 2048);
  draft_options(args, mo);
  args.finish();
  if (mo.draft_model.empty())
    fail(ErrorKind::kInvalidArgument, "--prompts measures speculative decoding: pass --draft F");
  check_conditions(allow_battery, allow_non_release);
  std::ifstream in(path_from_utf8(prompts_path), std::ios::binary);
  if (!in) fail(ErrorKind::kIo, "cannot read '{}'", prompts_path);
  std::vector<std::string> prompts;
  for (std::string line; std::getline(in, line);) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (!line.empty()) prompts.push_back(line);
  }
  if (prompts.empty()) fail(ErrorKind::kInvalidArgument, "'{}' holds no prompt", prompts_path);

  const PerformanceMode perf;
  const Environment env = capture_environment(500);
  Session s(mo);
  model::LlamaRef& draft = *detail::EngineAccess::draft(s.engine);
  std::vector<std::vector<std::int32_t>> ids;
  for (const std::string& p : prompts) {
    const ChatMessage m{"user", p};
    ids.push_back(s.engine.chat_prompt("", std::span(&m, 1), "2026-11-12"));
  }
  model::SpeculativeState costs;  // as an Engine keeps them across turns
  struct Run {
    model::GenerateResult r;
    std::vector<std::int32_t> out;
  };
  const bool greedy = temp <= 0.0f;
  auto once = [&](std::size_t p, bool speculative, int n, std::uint64_t run_seed) {
    Run run;
    s.llm.reset();
    draft.reset();
    costs.forget_context();
    sampling::Sampler sampler({temp, top_k, top_p, run_seed});
    const auto emit = [&](std::int32_t id) { run.out.push_back(id); };
    run.r = speculative
                ? model::generate_speculative(s.llm, draft, s.tok, sampler, ids[p], n, mo.speculative, emit, &costs)
                : model::generate(s.llm, s.tok, sampler, ids[p], n, emit);
    return run;
  };
  std::printf("bench: %s, %s, %d threads, %s build, power %s\n", mo.model.c_str(), mo.backend.c_str(), mo.threads,
              build_info().build_type, to_string(power_source()));
  const std::string how =
      greedy ? std::string("greedy")
             : std::format("temperature {}, top-k {}, top-p {}, seed {} + 1000 rep + prompt", temp, top_k, top_p, seed);
  std::printf("speculative: %s; %zu prompts from %s, %s, up to %d tokens, %d rep(s), order offset %d\n",
              s.engine.draft_description().c_str(), prompts.size(), prompts_path.c_str(), how.c_str(), gen, reps,
              round_offset);
  std::printf("environment: %s; %s; power mode %s; commit %s; %.1f GiB RAM available\n", env.os.c_str(),
              env.cpu.c_str(), env.power_overlay.c_str(), env.commit.c_str(), env.ram_available_gib);
  for (const BusyProcess& b : env.busy) std::printf("busy: %s %.1f %% of the machine\n", b.name.c_str(), b.cpu_percent);
  once(0, false, 16, seed);  // warm-up, unmeasured
  once(0, true, 16, seed);
  std::printf("%-4s %-6s %-5s %7s %10s %10s %7s %9s %s\n", "rep", "prompt", "first", "tokens", "plain t/s", "spec t/s",
              "ratio", "kept", "identical");
  std::vector<double> ratios, plain_rates, spec_rates;
  double plain_tokens = 0, plain_time = 0, spec_tokens = 0, spec_time = 0;
  long long drafted = 0, accepted = 0, looked_up = 0;
  int mismatches = 0;
  for (int rep = 0; rep < reps; ++rep) {
    for (std::size_t p = 0; p < prompts.size(); ++p) {
      const bool spec_first = (static_cast<std::size_t>(rep + round_offset) + p) % 2 == 1;
      Run a, b;  // a: plain, b: speculative
      const std::uint64_t run_seed = seed + 1000 * static_cast<std::uint64_t>(rep) + p;
      if (spec_first) {
        b = once(p, true, gen, run_seed);
        a = once(p, false, gen, run_seed);
      } else {
        a = once(p, false, gen, run_seed);
        b = once(p, true, gen, run_seed);
      }
      // Greedy: the two must be identical. Sampled: equal in distribution only.
      const bool same = !greedy || (a.out == b.out && a.r.stop == b.r.stop);
      mismatches += same ? 0 : 1;
      const double ra = a.r.generated / a.r.decode_seconds, rb = b.r.generated / b.r.decode_seconds;
      ratios.push_back(rb / ra);
      plain_rates.push_back(ra);
      spec_rates.push_back(rb);
      plain_tokens += a.r.generated;
      plain_time += a.r.decode_seconds;
      spec_tokens += b.r.generated;
      spec_time += b.r.decode_seconds;
      drafted += b.r.drafted;
      accepted += b.r.accepted;
      looked_up += b.r.looked_up;
      const std::string kept = std::format("{}/{}", b.r.accepted, b.r.drafted);
      std::printf("%-4d %-6zu %-5s %7d %10.2f %10.2f %7.3f %9s %s\n", rep, p, spec_first ? "spec" : "plain",
                  a.r.generated, ra, rb, rb / ra, kept.c_str(),
                  !greedy ? "n/a"
                  : same  ? "yes"
                          : "NO");
      std::fflush(stdout);
    }
  }
  const Stats rs = stats(ratios), ps = stats(plain_rates), ss = stats(spec_rates);
  double log_sum = 0;
  for (const double r : ratios) log_sum += std::log(r);
  std::printf("plain decode tok/s: median %.2f (IQR %.2f .. %.2f); all tokens / all time %.2f\n", ps.median, ps.p25,
              ps.p75, plain_tokens / plain_time);
  std::printf("speculative decode tok/s: median %.2f (IQR %.2f .. %.2f); all tokens / all time %.2f\n", ss.median,
              ss.p25, ss.p75, spec_tokens / spec_time);
  std::printf("speedup per run: median %.3fx (IQR %.3f .. %.3f), geometric mean %.3fx; all tokens / all time %.3fx\n",
              rs.median, rs.p25, rs.p75, std::exp(log_sum / static_cast<double>(ratios.size())),
              (spec_tokens / spec_time) / (plain_tokens / plain_time));
  if (greedy) {
    std::printf(
        "draft tokens kept: %lld of %lld (%.1f %%; %lld proposed by prompt lookup); identical output in %zu of %zu "
        "runs\n",
        accepted, drafted, 100.0 * static_cast<double>(accepted) / static_cast<double>(std::max(drafted, 1LL)),
        looked_up, ratios.size() - static_cast<std::size_t>(mismatches), ratios.size());
  } else {
    std::printf("draft tokens kept: %lld of %lld (%.1f %%); sampled, so the texts differ (same distribution)\n",
                accepted, drafted, 100.0 * static_cast<double>(accepted) / static_cast<double>(std::max(drafted, 1LL)));
  }
  std::printf("measured costs: draft step %.1f ms, target step %.1f ms, extra token %.1f ms, threshold %.3f\n",
              costs.draft_step() * 1e3, costs.target_step() * 1e3, costs.extra_token() * 1e3, costs.threshold());
  if (power_source() != PowerSource::kAc && !allow_battery) {
    fail(ErrorKind::kInvalidArgument,
         "the machine left AC power during the measurement; the numbers above are not valid (plug in and rerun)");
  }
  if (mismatches > 0) {
    fail(ErrorKind::kInternal, "speculative decoding changed the greedy output in {} run(s)", mismatches);
  }
  return 0;
}

}  // namespace

int cmd_bench(Args& args) {
  if (const auto prompts = args.value("prompts")) return bench_speculative(args, *prompts);
  const int prompt = static_cast<int>(args.int_or("prompt", 512, 0, 1 << 16));
  const int gen = static_cast<int>(args.int_or("gen", 128, 0, 1 << 16));
  const int reps = static_cast<int>(args.int_or("reps", 5, 1, 100));
  const bool allow_battery = args.flag("allow-battery");
  const bool allow_non_release = args.flag("allow-non-release");
  const double bandwidth = args.double_or("bandwidth", 0.0, 0.0, 1e4);
  const bool with_counters = args.flag("counters");
  const ModelOptions mo = model_options(args, static_cast<std::uint32_t>(std::max({prompt, gen, 1})));
  args.finish();
  if (prompt == 0 && gen == 0) fail(ErrorKind::kInvalidArgument, "nothing to measure: --prompt and --gen are both 0");
  const bool release = build_info().build_type == std::string("Release");
  if (!release && !allow_non_release) {
    fail(ErrorKind::kInvalidArgument,
         "bench measures performance; use a Release build (this is {}; --allow-non-release runs it anyway)",
         build_info().build_type);
  }
  if (power_source() != PowerSource::kAc && !allow_battery) {
    fail(ErrorKind::kInvalidArgument,
         "the machine is not on AC power; battery limits clocks and makes numbers incomparable "
         "(pass --allow-battery to measure anyway)");
  }

  // Measurement hygiene: no sleep, no EcoQoS throttling, and a record of
  // what else was running.
  const PerformanceMode perf;
  const Environment env = capture_environment(500);
  Session s(mo);
  const std::uint32_t n_vocab = s.llm.config().n_vocab;
  std::vector<float> logits(n_vocab);
  // llama-bench's inputs: BOS first, then pseudo-random token IDs. A
  // multiplicative hash spreads them over the vocabulary, the same on every run.
  std::vector<std::int32_t> tokens(static_cast<std::size_t>(std::max(prompt, gen)));
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    tokens[i] = static_cast<std::int32_t>((static_cast<std::uint64_t>(i + 1) * 2654435761u) % n_vocab);
  }
  if (!tokens.empty() && s.tok.bos() >= 0) tokens[0] = s.tok.bos();

  using Clock = std::chrono::steady_clock;
  auto pp = [&] {
    s.llm.reset();
    const auto t0 = Clock::now();
    s.llm.forward(std::span(tokens).first(static_cast<std::size_t>(prompt)), logits, false);
    return std::chrono::duration<double>(Clock::now() - t0).count();
  };
  auto tg = [&] {
    s.llm.reset();
    const auto t0 = Clock::now();
    for (int i = 0; i < gen; ++i)
      s.llm.forward(std::span(tokens).subspan(static_cast<std::size_t>(i), 1), logits, false);
    return std::chrono::duration<double>(Clock::now() - t0).count();
  };

  std::printf("bench: %s, %s, %d threads, %d reps after 1 warm-up, %s build, power %s\n", mo.model.c_str(),
              mo.backend.c_str(), mo.threads, reps, build_info().build_type, to_string(power_source()));
  std::printf("environment: %s; %s; power mode %s; commit %s; %.1f GiB RAM available\n", env.os.c_str(),
              env.cpu.c_str(), env.power_overlay.c_str(), env.commit.c_str(), env.ram_available_gib);
  for (const BusyProcess& b : env.busy) std::printf("busy: %s %.1f %% of the machine\n", b.name.c_str(), b.cpu_percent);
  std::printf("%-8s %14s %24s\n", "test", "tok/s median", "IQR (25th..75th pct)");
  std::vector<std::string> counter_lines;
  auto measure = [&](const char* name, int n, auto&& once) {
    if (n == 0) return Stats{0, 0, 0};
    once();  // warm-up
    std::vector<double> rate;
    rate.reserve(static_cast<std::size_t>(reps));
    Counters counters;
    if (with_counters) counters.start();
    for (int r = 0; r < reps; ++r) rate.push_back(n / once());
    const Counters::Medians cm = with_counters ? counters.stop() : Counters::Medians{};
    const Stats st = stats(rate);
    const std::string test = std::format("{}{}", name, n);
    std::printf("%-8s %14.2f %11.2f .. %-10.2f\n", test.c_str(), st.median, st.p25, st.p75);
    std::fflush(stdout);
    if (with_counters) counter_lines.push_back(std::format("counters {}: {}", test, format_counters(cm)));
    return st;
  };
  measure("pp", prompt, pp);
  const Stats g = measure("tg", gen, tg);
  for (const std::string& line : counter_lines) std::printf("%s\n", line.c_str());
  if (gen > 0) {
    const double bytes = static_cast<double>(model::decode_weight_bytes(s.file));
    const double gbps = bytes * g.median / 1e9;
    std::printf("decode reads %.3f GB of weights per token: %.1f GB/s at the median tg rate\n", bytes / 1e9, gbps);
    if (bandwidth > 0.0 && gbps > 1.05 * bandwidth) {
      fail(ErrorKind::kInternal,
           "decode would read {:.1f} GB/s, more than 1.05 x the measured {:.1f} GB/s: the measurement is wrong", gbps,
           bandwidth);
    }
  }
  // The power source is checked again: a machine unplugged during the run
  // lowers its clocks partway, and the numbers above would mix two states.
  if (power_source() != PowerSource::kAc && !allow_battery) {
    fail(ErrorKind::kInvalidArgument,
         "the machine left AC power during the measurement; the numbers above are not valid (plug in and rerun)");
  }
  if (!release)
    std::printf("note: %s build; these numbers are not comparable to Release measurements\n", build_info().build_type);
  return 0;
}

}  // namespace hearth::cli
