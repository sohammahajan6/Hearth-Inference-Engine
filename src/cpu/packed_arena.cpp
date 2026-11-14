#include "cpu/packed_arena.h"

#include "core/error.h"

namespace hearth::cpu {
namespace {

std::size_t align_up(std::size_t x, std::size_t a) { return (x + a - 1) / a * a; }

}  // namespace

PackedArena::PackedArena(std::span<const MatrixShape> shapes, std::size_t max_slice)
    : shapes_(shapes.begin(), shapes.end()), packed_(shapes.size(), false) {
  HEARTH_CHECK(max_slice >= kAlign && max_slice % kAlign == 0);
  std::vector<std::size_t> slice_end;  // bytes used in each slice so far
  for (std::size_t i = 0; i < shapes_.size(); ++i) {
    const MatrixShape& s = shapes_[i];
    const std::size_t bytes = packed_bytes(s.type, s.rows, s.k);
    if (bytes > max_slice) {
      fail(ErrorKind::kInvalidArgument, "packed arena: matrix {} ({} x {}) needs {} bytes, more than a slice ({})", i,
           s.rows, s.k, bytes, max_slice);
    }
    std::size_t offset = slice_end.empty() ? max_slice : align_up(slice_end.back(), kAlign);
    if (offset + bytes > max_slice) {  // cut between matrices: start a new slice
      slice_end.push_back(0);
      offset = 0;
    }
    locations_.push_back({slice_end.size() - 1, offset, bytes, packed_quant_bytes(s.type, s.rows, s.k)});
    slice_end.back() = offset + bytes;
  }
  for (const std::size_t end : slice_end) slices_.emplace_back(align_up(end, kAlign));  // zero-filled by the OS
}

void PackedArena::pack(std::size_t i, const void* blocks) {
  HEARTH_CHECK(i < shapes_.size() && !packed_[i]);
  const MatrixShape& s = shapes_[i];
  const Location& l = locations_[i];
  pack_matrix(s.type, blocks, s.rows, s.k, slices_[l.slice].data() + l.offset);
  packed_[i] = true;
}

PackedView PackedArena::view(std::size_t i) const {
  HEARTH_CHECK(i < shapes_.size() && packed_[i]);
  const MatrixShape& s = shapes_[i];
  const Location& l = locations_[i];
  const std::byte* base = slices_[l.slice].data() + l.offset;
  return PackedView{s.type,
                    s.rows,
                    s.k,
                    groups_for(s.k),
                    reinterpret_cast<const std::uint8_t*>(base),
                    reinterpret_cast<const std::uint16_t*>(base + l.quant_bytes)};
}

std::size_t PackedArena::bytes() const noexcept {
  std::size_t n = 0;
  for (const PageBuffer& b : slices_) n += b.size();
  return n;
}

}  // namespace hearth::cpu
