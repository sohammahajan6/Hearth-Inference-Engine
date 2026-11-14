#include "core/trace.h"

#include <windows.h>

#include <algorithm>
#include <atomic>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "core/clock.h"
#include "core/error.h"
#include "core/json_writer.h"
#include "core/platform.h"

namespace hearth::trace {
namespace {

std::atomic<bool> g_on{false};
std::mutex g_mutex;
std::vector<Event> g_events;
std::map<std::uint32_t, std::string> g_lane_names;
std::atomic<std::uint64_t> g_dropped{0};

}  // namespace

void start() {
  const std::lock_guard lock(g_mutex);
  g_events.clear();
  g_dropped.store(0, std::memory_order_relaxed);
  g_on.store(true, std::memory_order_relaxed);
}

bool on() noexcept { return g_on.load(std::memory_order_relaxed); }

void record_on(std::uint32_t lane, const char* name, std::int64_t t0_ns, std::int64_t t1_ns, int layer,
               std::int64_t rows, std::int64_t tokens) {
  const std::lock_guard lock(g_mutex);
  if (g_events.size() >= kMaxEvents) {
    g_dropped.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  g_events.push_back({name, t0_ns, t1_ns, rows, tokens, layer, lane});
}

void record(const char* name, std::int64_t t0_ns, std::int64_t t1_ns, int layer, std::int64_t rows,
            std::int64_t tokens) {
  record_on(GetCurrentThreadId(), name, t0_ns, t1_ns, layer, rows, tokens);
}

void name_lane(std::uint32_t lane, std::string_view title) {
  const std::lock_guard lock(g_mutex);
  g_lane_names[lane] = std::string(title);
}

std::vector<Event> events() {
  const std::lock_guard lock(g_mutex);
  return g_events;
}

void write(const std::filesystem::path& path) {
  g_on.store(false, std::memory_order_relaxed);
  const std::lock_guard lock(g_mutex);
  // Events are stored when they end, so the earliest start is not always first.
  std::int64_t origin = g_events.empty() ? 0 : g_events.front().t0_ns;
  for (const Event& e : g_events) origin = std::min(origin, e.t0_ns);
  JsonWriter w(false);
  w.begin_object();
  w.field("displayTimeUnit", "ms");
  w.key("traceEvents");
  w.begin_array();
  // Metadata events ("ph": "M") title the named lanes and list them first.
  for (const auto& [lane, title] : g_lane_names) {
    w.begin_object();
    w.field("name", "thread_name");
    w.field("ph", "M");
    w.field("pid", 1);
    w.field("tid", lane);
    w.key("args");
    w.begin_object();
    w.field("name", title);
    w.end_object();
    w.end_object();
    w.begin_object();
    w.field("name", "thread_sort_index");
    w.field("ph", "M");
    w.field("pid", 1);
    w.field("tid", lane);
    w.key("args");
    w.begin_object();
    w.field("sort_index", -1);
    w.end_object();
    w.end_object();
  }
  for (const Event& e : g_events) {
    w.begin_object();
    w.field("name", e.name);
    w.field("ph", "X");
    w.field("pid", 1);
    w.field("tid", e.lane);
    // Microseconds, as the format requires; fractional values keep QPC's 100 ns resolution.
    w.field("ts", static_cast<double>(e.t0_ns - origin) / 1e3);
    w.field("dur", static_cast<double>(e.t1_ns - e.t0_ns) / 1e3);
    w.key("args");
    w.begin_object();
    if (e.layer >= 0) w.field("layer", e.layer);
    if (e.rows > 0) w.field("rows", e.rows);
    if (e.tokens > 0) w.field("tokens", e.tokens);
    w.end_object();
    w.end_object();
  }
  w.end_array();
  w.key("otherData");
  w.begin_object();
  w.field("producer", "hearth");
  w.field("dropped_events", g_dropped.load(std::memory_order_relaxed));
  w.end_object();
  w.end_object();
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out << w.str() << "\n";
  out.close();
  if (!out) fail(ErrorKind::kIo, "cannot write the trace to {}", path_to_utf8(path));
}

Scope::Scope(const char* name, int layer, std::int64_t rows, std::int64_t tokens) noexcept
    : name_(name), layer_(layer), rows_(rows), tokens_(tokens) {
  if (on()) t0_ = now_ns();
}

Scope::~Scope() {
  if (t0_ >= 0 && on()) {
    // Recording must never break the computation it observes: an event that
    // cannot be stored (out of memory) is counted as dropped.
    try {
      record(name_, t0_, now_ns(), layer_, rows_, tokens_);
    } catch (...) {
      g_dropped.fetch_add(1, std::memory_order_relaxed);
    }
  }
}

}  // namespace hearth::trace
