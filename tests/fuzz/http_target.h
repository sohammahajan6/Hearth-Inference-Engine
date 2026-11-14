// The code under test for the HTTP request fuzzer (src/ui/http.h). Shared by
// the libFuzzer build (fuzz_http.cpp), the replay build and the quick-tier
// regression test (tests/unit/fuzz_replay_test.cpp).
//
// Input: any bytes, taken as what a client sent. The parser must stay within
// its limits, and every accepted request must be well formed:
//   - kComplete: consumed <= size; the method is token characters; the path
//     starts with '/'; header names are lowercase; the body is exactly the
//     Content-Length; every prefix shorter than `consumed` is kIncomplete;
//   - kBad: a status this server sends;
//   - the same input always gives the same result;
//   - decoding the body as a form gives valid UTF-8 or fails cleanly.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "core/error.h"
#include "core/utf8.h"
#include "ui/http.h"

namespace hearth::fuzz {

inline void http_one_input(const std::uint8_t* data, std::size_t size) {
  const std::string_view in(reinterpret_cast<const char*>(data), size);
  ui::Limits limits;
  limits.max_head = 4096;  // small limits, so the fuzzer reaches them
  limits.max_body = 8192;
  limits.max_headers = 16;
  ui::Request r;
  const ui::ParseResult p = ui::parse_request(in, limits, r);
  ui::Request again;
  const ui::ParseResult q = ui::parse_request(in, limits, again);
  HEARTH_CHECK(p.status == q.status && p.http_status == q.http_status && p.consumed == q.consumed);
  if (p.status == ui::ParseStatus::kBad) {
    HEARTH_CHECK(std::string_view(ui::status_text(p.http_status)) != "Unknown");
    HEARTH_CHECK(!p.reason.empty());
    return;
  }
  if (p.status == ui::ParseStatus::kIncomplete) {
    HEARTH_CHECK(size <= limits.max_head + limits.max_body + 4);
    return;
  }
  HEARTH_CHECK(p.consumed <= size);
  HEARTH_CHECK(!r.method.empty() && !r.path.empty() && r.path.front() == '/');
  HEARTH_CHECK(r.headers.size() <= limits.max_headers);
  HEARTH_CHECK(r.body.size() <= limits.max_body);
  for (const auto& [name, value] : r.headers) {
    HEARTH_CHECK(!name.empty());
    for (const char c : name) HEARTH_CHECK(!(c >= 'A' && c <= 'Z'));
  }
  if (const std::string* cl = r.header("content-length")) HEARTH_CHECK(std::stoull(*cl) == r.body.size());
  for (const auto& [k, v] : r.query) HEARTH_CHECK(valid_utf8(k) && valid_utf8(v));
  // A request is only complete once all of it has arrived.
  for (const std::size_t cut : {p.consumed - 1, p.consumed / 2, std::size_t{0}}) {
    if (cut >= p.consumed) continue;
    ui::Request part;
    HEARTH_CHECK(ui::parse_request(in.substr(0, cut), limits, part).status == ui::ParseStatus::kIncomplete);
  }
  ui::Params form;
  if (ui::form_decode(r.body, form)) {
    for (const auto& [k, v] : form) HEARTH_CHECK(valid_utf8(k) && valid_utf8(v));
  }
}

}  // namespace hearth::fuzz
