// Exact arithmetic for the GPU activation quantizer (quantize_act.comp) and
// its test shader (probe_div_exact.comp); docs/architecture/08-gpu-backend.md
// section 2, "Activations". Only integer operations, so the results do not
// depend on the precision Vulkan allows for division (2.5 ulp) or on the
// denormal mode.

// a / b, correctly rounded to nearest even, for positive normal a and b whose
// quotient is a normal float (the quantizer's domain guarantees both). The
// 24-bit significands are divided bit by bit (restoring long division); the
// remainder stays below 2^25, so 32-bit integers suffice (no shaderInt64).
float div_exact(float a, float b) {
  const uint ua = floatBitsToUint(a);
  const uint ub = floatBitsToUint(b);
  int e = int((ua >> 23) & 0xFFu) - int((ub >> 23) & 0xFFu);
  uint ma = (ua & 0x7FFFFFu) | 0x800000u;
  const uint mb = (ub & 0x7FFFFFu) | 0x800000u;
  if (ma < mb) {  // make the significand quotient lie in [1, 2)
    ma <<= 1;
    e -= 1;
  }
  uint q = 0u;
  uint rem = ma;
  for (int i = 0; i < 24; ++i) {
    q <<= 1;
    if (rem >= mb) {
      rem -= mb;
      q |= 1u;
    }
    rem <<= 1;
  }
  // rem is now twice the remainder: compare it with mb for the rounding bit.
  if (rem > mb || (rem == mb && (q & 1u) != 0u)) q += 1u;
  if (q == 0x1000000u) {  // rounded up to 2: renormalize
    q >>= 1;
    e += 1;
  }
  return uintBitsToFloat((uint(e + 127) << 23) | (q & 0x7FFFFFu));
}

// fp32 to fp16 bits, round to nearest even: core/fp16.h fp32_to_fp16 for
// non-negative finite inputs, with the subnormal case done in integers too.
uint float_to_half_bits(float f) {
  uint x = floatBitsToUint(f) & 0x7FFFFFFFu;
  if (x >= 0x47800000u) return 0x7C00u;  // >= 65536 (or inf): infinity
  if (x < 0x38800000u) {                 // below 2^-14: fp16 subnormal or zero
    const uint e = x >> 23;
    if (e < 102u) return 0u;  // below 2^-25: rounds to zero
    const uint m = (x & 0x7FFFFFu) | 0x800000u;
    const uint shift = 126u - e;  // 14 .. 24: m * 2^-shift units of 2^-24
    uint q = m >> shift;
    const uint rem = m & ((1u << shift) - 1u);
    const uint half_ = 1u << (shift - 1u);
    if (rem > half_ || (rem == half_ && (q & 1u) != 0u)) q += 1u;
    return q;
  }
  const uint mant_odd = (x >> 13) & 1u;
  x += 0xC8000FFFu + mant_odd;  // rebias 127 -> 15 and add the rounding bias
  return x >> 13;
}

// fp16 bits to float, exact for every input and every denormal mode.
float half_bits_to_float_q(uint h) {
  const uint s = (h & 0x8000u) << 16;
  const uint e = (h >> 10) & 0x1Fu;
  const uint m = h & 0x3FFu;
  if (e == 0u) return uintBitsToFloat(floatBitsToUint(float(m) * 5.9604644775390625e-8) | s);
  if (e == 31u) return uintBitsToFloat(s | (m == 0u ? 0x7F800000u : (0x7FC00000u | (m << 13))));
  return uintBitsToFloat(s | ((e + 112u) << 23) | (m << 13));
}
