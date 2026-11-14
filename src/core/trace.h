// Timed events in the Chrome trace format (the "Trace Event Format" JSON),
// viewable in the free Perfetto UI (https://ui.perfetto.dev) or
// chrome://tracing: `hearth <command> --trace out.json`.
//
// Each Scope becomes one complete event ("ph": "X") on the thread that
// created it, with its name, start, duration, and the layer and shape it
// worked on. Recording is off unless start() was called; a Scope then costs
// one relaxed atomic load and a branch.
//
// Work that runs on the GPU is recorded on a lane of its own (kGpuLane): its
// times come from GPU timestamps mapped onto now_ns()'s clock
// (gpu/timestamps.h), so CPU threads and the GPU share one time axis.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string_view>
#include <vector>

namespace hearth::trace {

// The GPU's lane, shown next to the CPU threads. Windows thread IDs are
// multiples of 4, so an odd lane ID cannot be taken by a real thread.
inline constexpr std::uint32_t kGpuLane = 0xFFFF'0001u;

// Begins recording (and drops earlier events). At most kMaxEvents are kept;
// later ones are counted, and write() reports how many were dropped.
inline constexpr std::size_t kMaxEvents = std::size_t{1} << 22;
void start();
bool on() noexcept;
// Stops recording and writes the events. Throws Error(kIo) if the file cannot be written.
void write(const std::filesystem::path& path);

// `layer` < 0: not a per-layer operation. `rows` and `tokens` describe the
// operation's shape (0 = not applicable).
void record(const char* name, std::int64_t t0_ns, std::int64_t t1_ns, int layer, std::int64_t rows,
            std::int64_t tokens);
// The same on `lane` instead of the calling thread. `name` must outlive the
// recording (a string literal).
void record_on(std::uint32_t lane, const char* name, std::int64_t t0_ns, std::int64_t t1_ns, int layer,
               std::int64_t rows, std::int64_t tokens);
// The lane's title in the viewer. Kept across start() calls.
void name_lane(std::uint32_t lane, std::string_view title);

// A copy of the events recorded since start(), in the order they were stored
// (when they ended). For tests.
struct Event {
  const char* name;
  std::int64_t t0_ns, t1_ns, rows, tokens;
  int layer;
  std::uint32_t lane;  // the recording thread's ID, or a lane such as kGpuLane
};
std::vector<Event> events();

class Scope {
 public:
  explicit Scope(const char* name, int layer = -1, std::int64_t rows = 0, std::int64_t tokens = 0) noexcept;
  ~Scope();
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;

 private:
  const char* name_;
  int layer_;
  std::int64_t rows_, tokens_;
  std::int64_t t0_ = -1;  // -1: recording was off when the scope began
};

}  // namespace hearth::trace
