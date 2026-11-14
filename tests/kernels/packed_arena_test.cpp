// cpu::PackedArena: every matrix 4 KiB-aligned, slices cut only between
// matrices and never larger than the limit, and each matrix's bytes equal
// those of a stand-alone cpu::PackedMatrix (so the kernels' bits cannot change).
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "core/error.h"
#include "core/quant.h"
#include "cpu/packed_arena.h"
#include "cpu/qgemm.h"

using namespace hearth;

namespace {

std::vector<std::uint8_t> random_blocks(cpu::QType type, std::int64_t rows, std::int64_t k, std::mt19937& rng) {
  const std::size_t block = type == cpu::QType::kQ8_0 ? sizeof(quant::BlockQ8_0) : sizeof(quant::BlockQ4_0);
  std::vector<std::uint8_t> b(static_cast<std::size_t>(rows * (k / quant::kBlock)) * block);
  for (auto& x : b) x = static_cast<std::uint8_t>(rng());
  return b;
}

}  // namespace

TEST_SUITE("quick") {
  TEST_CASE("packed arena: aligned matrices, slices cut between them, bytes equal PackedMatrix") {
    using cpu::QType;
    const std::vector<cpu::MatrixShape> shapes = {{QType::kQ8_0, 3, 480},   {QType::kQ4_0, 5, 544},
                                                  {QType::kQ8_0, 9, 512},   {QType::kQ4_0, 1, 32},
                                                  {QType::kQ8_0, 13, 2048}, {QType::kQ4_0, 7, 8192}};
    constexpr std::size_t kSlice = 10 * cpu::PackedArena::kAlign;  // small, to force several slices
    cpu::PackedArena arena(shapes, kSlice);
    REQUIRE(arena.size() == shapes.size());
    std::mt19937 rng(3);
    std::vector<std::vector<std::uint8_t>> blocks;
    for (std::size_t i = 0; i < shapes.size(); ++i) {
      blocks.push_back(random_blocks(shapes[i].type, shapes[i].rows, shapes[i].k, rng));
      arena.pack(i, blocks.back().data());
    }
    std::size_t prev_slice = 0, prev_end = 0;
    for (std::size_t i = 0; i < shapes.size(); ++i) {
      CAPTURE(i);
      const auto& l = arena.location(i);
      CHECK(l.offset % cpu::PackedArena::kAlign == 0);
      CHECK(l.bytes == cpu::packed_bytes(shapes[i].type, shapes[i].rows, shapes[i].k));
      CHECK(l.offset + l.bytes <= arena.slice(l.slice).size());
      CHECK(arena.slice(l.slice).size() <= kSlice);
      // In order: the same slice after the previous matrix, or the next slice from offset 0.
      if (i > 0)
        CHECK(((l.slice == prev_slice && l.offset >= prev_end) || (l.slice == prev_slice + 1 && l.offset == 0)));
      prev_slice = l.slice;
      prev_end = l.offset + l.bytes;
      const cpu::PackedMatrix alone(shapes[i].type, blocks[i].data(), shapes[i].rows, shapes[i].k);
      const cpu::PackedView a = arena.view(i), b = alone.view();
      CHECK(a.rows == b.rows);
      CHECK(a.k == b.k);
      CHECK(a.groups == b.groups);
      CHECK(std::memcmp(a.q, b.q, l.quant_bytes) == 0);
      CHECK(std::memcmp(a.d, b.d, l.bytes - l.quant_bytes) == 0);
      CHECK(reinterpret_cast<const std::byte*>(a.q) == arena.slice(l.slice).data() + l.offset);
    }
    CHECK(arena.slices() >= 2);
  }

  TEST_CASE("packed arena: a matrix larger than a slice is a clear error") {
    const std::vector<cpu::MatrixShape> shapes = {{cpu::QType::kQ8_0, 64, 2048}};  // 69,632 bytes
    CHECK_THROWS_AS(cpu::PackedArena(shapes, 4 * cpu::PackedArena::kAlign), Error);
  }
}
