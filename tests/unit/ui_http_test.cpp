// src/ui: the HTTP request parser's rules and limits, form and percent
// decoding, the token comparison, and command-line quoting (checked by
// parsing the result back with Windows' own CommandLineToArgvW).
#include <doctest/doctest.h>

#include <windows.h>
//
#include <shellapi.h>

#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "ui/command_line.h"
#include "ui/http.h"

using namespace hearth;
using ui::ParseStatus;

namespace {

ui::ParseResult parse(std::string_view s, ui::Request& r, const ui::Limits& l = {}) {
  return ui::parse_request(s, l, r);
}

int bad_status(std::string_view s, const ui::Limits& l = {}) {
  ui::Request r;
  const ui::ParseResult p = parse(s, r, l);
  return p.status == ParseStatus::kBad ? p.http_status : 0;
}

}  // namespace

TEST_SUITE("quick") {
  TEST_CASE("ui http: a GET with a query, and a POST whose body arrives in parts") {
    ui::Request r;
    const std::string get = "GET /?t=abc%20d&x=1+2 HTTP/1.1\r\nHost: 127.0.0.1:5\r\nX-Hearth-Token:  tok \r\n\r\n";
    const ui::ParseResult p = parse(get, r);
    REQUIRE(p.status == ParseStatus::kComplete);
    CHECK(p.consumed == get.size());
    CHECK(r.method == "GET");
    CHECK(r.path == "/");
    REQUIRE(r.query_param("t") != nullptr);
    CHECK(*r.query_param("t") == "abc d");
    CHECK(*r.query_param("x") == "1 2");
    REQUIRE(r.header("x-hearth-token") != nullptr);
    CHECK(*r.header("x-hearth-token") == "tok");  // name lowercased, value trimmed
    CHECK(*r.header("host") == "127.0.0.1:5");

    const std::string post = "POST /api/chat HTTP/1.1\r\nHost: a\r\nContent-Length: 11\r\n\r\nmessage=hi!";
    for (std::size_t n = 0; n < post.size(); ++n) {
      ui::Request part;
      CAPTURE(n);
      CHECK(parse(post.substr(0, n), part).status == ParseStatus::kIncomplete);
    }
    ui::Request whole;
    const ui::ParseResult q = parse(post + "GET / HTTP/1.1\r\n", whole);  // a following request is not consumed
    REQUIRE(q.status == ParseStatus::kComplete);
    CHECK(q.consumed == post.size());
    CHECK(whole.body == "message=hi!");
  }

  TEST_CASE("ui http: malformed and oversized requests are refused with the right status") {
    CHECK(bad_status("GET / HTTP/1.1\r\n\r\n") == 400);                 // HTTP/1.1 without Host
    CHECK(bad_status("GET  / HTTP/1.1\r\nHost: a\r\n\r\n") == 400);     // double space
    CHECK(bad_status("GET / HTTP/1.1 x\r\nHost: a\r\n\r\n") == 400);    // extra field
    CHECK(bad_status("G(T / HTTP/1.1\r\nHost: a\r\n\r\n") == 400);      // not a token
    CHECK(bad_status("GET x HTTP/1.1\r\nHost: a\r\n\r\n") == 400);      // target not starting with /
    CHECK(bad_status("GET /a#b HTTP/1.1\r\nHost: a\r\n\r\n") == 400);   // fragment
    CHECK(bad_status("GET /\x01 HTTP/1.1\r\nHost: a\r\n\r\n") == 400);  // control character
    CHECK(bad_status("GET / HTTP/2.0\r\nHost: a\r\n\r\n") == 505);
    CHECK(bad_status("GET / HTTX/1.1\r\nHost: a\r\n\r\n") == 400);
    CHECK(bad_status("GET / HTTP/1.1\r\nHost: a\r\n folded\r\n\r\n") == 400);  // obsolete line folding
    CHECK(bad_status("GET / HTTP/1.1\r\nHost : a\r\n\r\n") == 400);            // space before the colon
    CHECK(bad_status("GET / HTTP/1.1\r\nHost: a\nX: b\r\n\r\n") == 400);       // bare LF
    CHECK(bad_status("GET / HTTP/1.1\r\nHost: a\rX: b\r\n\r\n") == 400);       // bare CR
    CHECK(bad_status("GET / HTTP/1.1\r\nHost: a\x7f\r\n\r\n") == 400);         // DEL in a value
    CHECK(bad_status("GET / HTTP/1.1\r\nHost: a\r\n: x\r\n\r\n") == 400);      // empty name
    CHECK(bad_status("POST / HTTP/1.1\r\nHost: a\r\nTransfer-Encoding: chunked\r\n\r\n") == 501);
    CHECK(bad_status("POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 1\r\nContent-Length: 1\r\n\r\nx") == 400);
    CHECK(bad_status("POST / HTTP/1.1\r\nHost: a\r\nContent-Length: -1\r\n\r\n") == 400);
    CHECK(bad_status("POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 1e3\r\n\r\n") == 400);
    CHECK(bad_status("POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 99999999999999\r\n\r\n") == 400);
    CHECK(bad_status("GET /?a=%zz HTTP/1.1\r\nHost: a\r\n\r\n") == 400);  // bad escape in the query
    CHECK(bad_status("GET /?a=%ff HTTP/1.1\r\nHost: a\r\n\r\n") == 400);  // not UTF-8
    CHECK(bad_status("\r\n\r\n") == 400);
    CHECK(bad_status("GET /" + std::string(5000, 'a') + " HTTP/1.1\r\nHost: a\r\n\r\n") == 414);

    ui::Limits small;
    small.max_head = 64;
    small.max_body = 4;
    small.max_headers = 2;
    CHECK(bad_status("GET / HTTP/1.1\r\nHost: a\r\nX: " + std::string(100, 'b') + "\r\n\r\n", small) == 431);
    ui::Request r;
    CHECK(parse(std::string(65, 'a'), r, small).http_status == 431);  // no end of head in sight: refused early
    CHECK(bad_status("POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 5\r\n\r\n", small) == 413);
    CHECK(bad_status("GET / HTTP/1.1\r\nHost: a\r\nA: 1\r\nB: 2\r\n\r\n", small) == 431);  // 3 headers
    // HTTP/1.0 needs no Host (the server refuses it later, for the Host check).
    CHECK(parse("GET / HTTP/1.0\r\n\r\n", r).status == ParseStatus::kComplete);
  }

  TEST_CASE("ui http: percent and form decoding, token comparison, status texts") {
    std::string out;
    CHECK(ui::percent_decode("a%20b+c", true, out));
    CHECK(out == "a b c");
    CHECK(ui::percent_decode("a+b", false, out));
    CHECK(out == "a+b");
    CHECK(ui::percent_decode("%E2%9C%94", true, out));
    CHECK(out == "\xE2\x9C\x94");
    CHECK_FALSE(ui::percent_decode("%", true, out));
    CHECK_FALSE(ui::percent_decode("%4", true, out));
    CHECK_FALSE(ui::percent_decode("%g0", true, out));
    CHECK_FALSE(ui::percent_decode("%C3", true, out));  // a truncated UTF-8 sequence
    CHECK_FALSE(ui::percent_decode("\xff", true, out));

    ui::Params p;
    REQUIRE(ui::form_decode("a=1&&b=x%20y&c&=v&d=", p));
    REQUIRE(p.size() == 5);
    CHECK(p[0].first == "a");
    CHECK(p[0].second == "1");
    CHECK(p[1].first == "b");
    CHECK(p[1].second == "x y");
    CHECK(p[2].first == "c");
    CHECK(p[2].second == "");
    CHECK(p[3].first == "");
    CHECK(p[3].second == "v");
    REQUIRE(ui::param(p, "d").has_value());
    CHECK(ui::param(p, "d")->empty());
    CHECK_FALSE(ui::param(p, "e").has_value());
    CHECK_FALSE(ui::form_decode("a=%zz", p));

    CHECK(ui::constant_time_equal("abc", "abc"));
    CHECK_FALSE(ui::constant_time_equal("abc", "abd"));
    CHECK_FALSE(ui::constant_time_equal("abc", "abcd"));
    CHECK(ui::constant_time_equal("", ""));
    for (const int code : {200, 400, 403, 404, 405, 408, 409, 413, 414, 421, 431, 500, 501, 503, 505}) {
      CAPTURE(code);
      CHECK(std::string_view(ui::status_text(code)) != "Unknown");
    }
  }

  TEST_CASE("ui command line: quoted arguments read back unchanged through CommandLineToArgvW") {
    const std::vector<std::wstring> args = {L"plain",
                                            L"with space",
                                            L"",
                                            L"quote\"inside",
                                            L"C:\\path with space\\",
                                            L"back\\\\slashes\\\"quote",
                                            L"E:\\Hetero-Engine-oak\\models\\Llama-3.2-1B-Instruct-Q8_0-pure.gguf",
                                            L"tab\there",
                                            L"\\\\server\\share\\x.gguf",
                                            L"\u00e9t\u00e9 \u2713"};
    std::wstring cmd = L"prog.exe";
    for (const std::wstring& a : args) cmd += L" " + ui::quote_argument(a);
    int argc = 0;
    LPWSTR* argv = CommandLineToArgvW(cmd.c_str(), &argc);
    REQUIRE(argv != nullptr);
    REQUIRE(argc == static_cast<int>(args.size()) + 1);
    for (std::size_t i = 0; i < args.size(); ++i) {
      CAPTURE(i);
      CHECK(std::wstring(argv[i + 1]) == args[i]);
    }
    LocalFree(argv);
    CHECK(ui::quote_argument(L"simple") == L"simple");
    CHECK(ui::quote_argument(L"a b") == L"\"a b\"");
  }
}
