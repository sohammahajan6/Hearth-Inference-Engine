// All packed weight matrices of a model in a few large page-aligned slices
//.
//
// The CPU kernels read each matrix through a PackedView exactly as before;
// the arena only decides where the bytes live. The GPU imports each slice
// once (one VkDeviceMemory and one VkBuffer, at most 4 GiB, the device's
// maxBufferSize) and binds each matrix as a range of it, so the weights exist
// once in RAM for both devices. Every matrix starts at a 4 KiB-aligned offset
// (the import granularity), and a slice is cut only between matrices.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "core/page_buffer.h"
#include "cpu/qgemm.h"

namespace hearth::cpu {

struct MatrixShape {
  QType type = QType::kQ8_0;
  std::int64_t rows = 0, k = 0;
};

class PackedArena {
 public:
  static constexpr std::size_t kAlign = 4096;
  static constexpr std::size_t kMaxSlice = std::size_t{4} << 30;  // the device's maxBufferSize

  // Where matrix i lives: slice, byte offset in it, and its size (the scales
  // start at offset + quant_bytes).
  struct Location {
    std::size_t slice = 0, offset = 0, bytes = 0, quant_bytes = 0;
  };

  PackedArena() = default;
  // Lays out the matrices in order and allocates the slices (zero-filled,
  // committed). Throws kInvalidArgument for a matrix larger than max_slice.
  explicit PackedArena(std::span<const MatrixShape> shapes, std::size_t max_slice = kMaxSlice);

  // Repacks matrix i from its GGUF blocks (rows * k / 32 blocks). Each matrix
  // is packed once.
  void pack(std::size_t i, const void* blocks);

  std::size_t size() const noexcept { return shapes_.size(); }
  PackedView view(std::size_t i) const;
  const Location& location(std::size_t i) const { return locations_.at(i); }
  std::size_t slices() const noexcept { return slices_.size(); }
  std::span<std::byte> slice(std::size_t s) const { return {slices_.at(s).data(), slices_.at(s).size()}; }
  std::size_t bytes() const noexcept;

 private:
  std::vector<MatrixShape> shapes_;
  std::vector<Location> locations_;
  std::vector<PageBuffer> slices_;
  std::vector<bool> packed_;
};

}  // namespace hearth::cpu
