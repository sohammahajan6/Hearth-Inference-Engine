// Windows performance counters (PDH), read-only.
//
// Used for the measurements logged next to every
// benchmark: CPU "% Processor Performance" (clock relative to base),
// per-process GPU engine utilisation and GPU memory, thermal zones. Counter
// paths are given in English and resolved with PdhAddEnglishCounter, so they
// work on any Windows display language.
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace hearth {

class PdhQuery {
 public:
  PdhQuery();
  ~PdhQuery();
  PdhQuery(const PdhQuery&) = delete;
  PdhQuery& operator=(const PdhQuery&) = delete;

  // Adds a counter (wildcard instances allowed, e.g. "\\GPU Engine(*)\\Utilization Percentage").
  // Returns its index, or -1 if this machine does not have the counter.
  int add(const std::wstring& english_path);

  // Takes a sample of every counter. Rate counters (utilisation) need two
  // samples before they have a value.
  void collect();

  // Current value of each instance of counter `index`: (instance name, value).
  std::vector<std::pair<std::string, double>> values(int index) const;

  // Sum of the instances whose name contains every string in `filters`.
  double sum(int index, const std::vector<std::string>& filters) const;
  // Largest value among the instances (e.g. the hottest thermal zone).
  double max(int index) const;

 private:
  void* query_ = nullptr;  // PDH_HQUERY
  std::vector<void*> counters_;
};

}  // namespace hearth
