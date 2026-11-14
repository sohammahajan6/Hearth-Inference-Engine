// Shared by the matvec_*.comp shaders.
//
// One workgroup of 256 invocations computes ROWS rows; each row gets
// 256 / ROWS invocations that walk its blocks with a stride and keep a float
// partial sum. The partials are then reduced in shared memory.

#extension GL_EXT_integer_dot_product : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require

layout(local_size_x = 256) in;
layout(constant_id = 0) const uint ROWS = 4;  // rows per workgroup (a power of two <= 256)

// Activations, quantized to Q8_0 by the host: 2 uvec4 (32 int8) per block,
// the decoded scale, and the sum of each block's quants.
layout(std430, set = 0, binding = 2) readonly buffer XQ { uvec4 xq[]; };
layout(std430, set = 0, binding = 3) readonly buffer XD { float xd[]; };
layout(std430, set = 0, binding = 4) readonly buffer XS { int xs[]; };
layout(std430, set = 0, binding = 5) writeonly buffer Y { float y[]; };

layout(push_constant) uniform Push {
  uint nb;         // blocks per row
  uint row_begin;  // first row of this dispatch
  uint row_end;    // one past the last row
  uint w_base0;    // element offset into binding 0 (rotating copies)
  uint w_base1;    // element offset into binding 1
} p;

shared float partial[256];

const uint LANES = 256u / ROWS;

uint row_index() { return p.row_begin + gl_WorkGroupID.x * ROWS + gl_LocalInvocationID.x / LANES; }
uint lane_index() { return gl_LocalInvocationID.x % LANES; }

// Q8_0 x Q8_0 on one block: 8 packed words of weights against the activation block.
int dot_q8(uvec4 w0, uvec4 w1, uint b) {
  const uvec4 x0 = xq[2u * b];
  const uvec4 x1 = xq[2u * b + 1u];
  int s = dotPacked4x8EXT(int(w0.x), int(x0.x));
  s += dotPacked4x8EXT(int(w0.y), int(x0.y));
  s += dotPacked4x8EXT(int(w0.z), int(x0.z));
  s += dotPacked4x8EXT(int(w0.w), int(x0.w));
  s += dotPacked4x8EXT(int(w1.x), int(x1.x));
  s += dotPacked4x8EXT(int(w1.y), int(x1.y));
  s += dotPacked4x8EXT(int(w1.z), int(x1.z));
  s += dotPacked4x8EXT(int(w1.w), int(x1.w));
  return s;
}

// Q4_0 x Q8_0 on one block: 4 packed words of nibbles (byte i holds elements
// i and i + 16). Unsigned nibbles times signed activations, minus 8 * sum(x).
int dot_q4(uvec4 q, uint b) {
  const uvec4 x0 = xq[2u * b];
  const uvec4 x1 = xq[2u * b + 1u];
  const uvec4 lo = q & 0x0F0F0F0Fu;
  const uvec4 hi = (q >> 4u) & 0x0F0F0F0Fu;
  int s = dotPacked4x8EXT(lo.x, int(x0.x));
  s += dotPacked4x8EXT(lo.y, int(x0.y));
  s += dotPacked4x8EXT(lo.z, int(x0.z));
  s += dotPacked4x8EXT(lo.w, int(x0.w));
  s += dotPacked4x8EXT(hi.x, int(x1.x));
  s += dotPacked4x8EXT(hi.y, int(x1.y));
  s += dotPacked4x8EXT(hi.z, int(x1.z));
  s += dotPacked4x8EXT(hi.w, int(x1.w));
  return s - 8 * xs[b];
}

float fp16_bits_to_float(uint16_t bits) { return unpackHalf2x16(uint(bits)).x; }

// Tree reduction of each row's LANES partials; lane 0 writes y[row].
void reduce_and_store(float acc) {
  const uint i = gl_LocalInvocationID.x;
  partial[i] = acc;
  barrier();
  for (uint stride = LANES / 2u; stride > 0u; stride >>= 1u) {
    if (lane_index() < stride) partial[i] += partial[i + stride];
    barrier();
  }
  const uint r = row_index();
  if (lane_index() == 0u && r < p.row_end) y[r] = partial[i];
}
