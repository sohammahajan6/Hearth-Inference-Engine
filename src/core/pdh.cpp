#include "core/pdh.h"

#include <windows.h>
// (separate block: pdh.h needs windows.h first)
#include <pdh.h>
#include <pdhmsg.h>

#include <algorithm>
#include <limits>

#include "core/error.h"

namespace hearth {
namespace {

std::string narrow(const wchar_t* w) {
  const int len = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
  if (len <= 1) return {};
  std::string s(static_cast<std::size_t>(len - 1), '\0');
  WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), len, nullptr, nullptr);
  return s;
}

}  // namespace

PdhQuery::PdhQuery() {
  PDH_HQUERY q = nullptr;
  const PDH_STATUS st = PdhOpenQueryW(nullptr, 0, &q);
  if (st != ERROR_SUCCESS) fail(ErrorKind::kIo, "PdhOpenQuery failed (status 0x{:08x})", static_cast<unsigned>(st));
  query_ = q;
}

PdhQuery::~PdhQuery() {
  if (query_ != nullptr) PdhCloseQuery(static_cast<PDH_HQUERY>(query_));
}

int PdhQuery::add(const std::wstring& english_path) {
  PDH_HCOUNTER c = nullptr;
  if (PdhAddEnglishCounterW(static_cast<PDH_HQUERY>(query_), english_path.c_str(), 0, &c) != ERROR_SUCCESS) return -1;
  counters_.push_back(c);
  return static_cast<int>(counters_.size() - 1);
}

void PdhQuery::collect() { PdhCollectQueryData(static_cast<PDH_HQUERY>(query_)); }

std::vector<std::pair<std::string, double>> PdhQuery::values(int index) const {
  std::vector<std::pair<std::string, double>> out;
  if (index < 0 || static_cast<std::size_t>(index) >= counters_.size()) return out;
  auto* c = static_cast<PDH_HCOUNTER>(counters_[static_cast<std::size_t>(index)]);
  DWORD bytes = 0, count = 0;
  PDH_STATUS st = PdhGetFormattedCounterArrayW(c, PDH_FMT_DOUBLE | PDH_FMT_NOCAP100, &bytes, &count, nullptr);
  if (st != static_cast<PDH_STATUS>(PDH_MORE_DATA) || bytes == 0) return out;
  std::vector<std::byte> buf(bytes);
  auto* items = reinterpret_cast<PDH_FMT_COUNTERVALUE_ITEM_W*>(buf.data());
  st = PdhGetFormattedCounterArrayW(c, PDH_FMT_DOUBLE | PDH_FMT_NOCAP100, &bytes, &count, items);
  if (st != ERROR_SUCCESS) return out;
  for (DWORD i = 0; i < count; ++i) {
    if (items[i].FmtValue.CStatus == ERROR_SUCCESS || items[i].FmtValue.CStatus == PDH_CSTATUS_NEW_DATA) {
      out.emplace_back(narrow(items[i].szName), items[i].FmtValue.doubleValue);
    }
  }
  return out;
}

double PdhQuery::sum(int index, const std::vector<std::string>& filters) const {
  double total = 0;
  for (const auto& [name, v] : values(index)) {
    if (std::all_of(filters.begin(), filters.end(),
                    [&](const std::string& f) { return name.find(f) != std::string::npos; })) {
      total += v;
    }
  }
  return total;
}

double PdhQuery::max(int index) const {
  double best = std::numeric_limits<double>::quiet_NaN();
  for (const auto& [name, v] : values(index)) {
    if (!(v <= best)) best = v;  // also replaces the initial NaN
  }
  return best;
}

}  // namespace hearth
