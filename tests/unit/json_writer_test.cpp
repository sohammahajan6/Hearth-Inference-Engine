#include <doctest/doctest.h>

#include <cstdint>
#include <limits>
#include <string>

#include "core/error.h"
#include "core/json_writer.h"
#include "core/page_buffer.h"

using hearth::JsonWriter;

TEST_SUITE("quick") {
  TEST_CASE("json: nested structure with commas placed correctly (compact)") {
    JsonWriter w(false);
    w.begin_object();
    w.field("name", "probe");
    w.field("n", 3);
    w.field("big", std::numeric_limits<std::uint64_t>::max());
    w.field("neg", std::int64_t{-42});
    w.field("ok", true);
    w.key("list");
    w.begin_array();
    w.value(1.5);
    w.value(0.1);
    w.null();
    w.begin_object();
    w.end_object();
    w.end_array();
    w.end_object();
    CHECK(w.str() ==
          R"({"name":"probe","n":3,"big":18446744073709551615,"neg":-42,"ok":true,"list":[1.5,0.1,null,{}]})");
  }

  TEST_CASE("json: escaping and non-finite numbers") {
    CHECK(JsonWriter::escape("a\"b\\c\n\t\x01") == "a\\\"b\\\\c\\n\\t\\u0001");
    CHECK(JsonWriter::escape("h\xC3\xA9llo") == "h\xC3\xA9llo");  // UTF-8 passes through
    JsonWriter w(false);
    w.begin_array();
    w.value(std::numeric_limits<double>::quiet_NaN());
    w.value(std::numeric_limits<double>::infinity());
    w.end_array();
    CHECK(w.str() == "[null,null]");
  }

  TEST_CASE("json: misuse is caught as an internal error") {
    JsonWriter w(false);
    w.begin_object();
    CHECK_THROWS_AS(w.value(1), hearth::Error);  // object member without a key
  }

  TEST_CASE("page buffer: page-aligned, zeroed, movable") {
    hearth::PageBuffer b(3 * hearth::PageBuffer::page_size() + 5);
    REQUIRE(b.data() != nullptr);
    CHECK(reinterpret_cast<std::uintptr_t>(b.data()) % hearth::PageBuffer::page_size() == 0);
    CHECK(b.data()[b.size() - 1] == std::byte{0});
    hearth::PageBuffer moved(std::move(b));
    CHECK(b.data() == nullptr);  // NOLINT(bugprone-use-after-move): checking the moved-from state is the point
    CHECK(moved.size() == 3 * hearth::PageBuffer::page_size() + 5);
  }
}
