
#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <vector>
#include <algorithm>
#include <bit>

// Proposed optimizations

inline int get_exponent_pow2(float x) {
    if (x == 0)
        return -1023; // Should not happen for power of 2
    uint64_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    return ((bits >> 52) & 0x7FF) - 1023;
}

inline int fast_ceil_log2(float x) {
    if (x <= 0)
        return -std::numeric_limits<float>::infinity(); // Or some error
    uint64_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    int exp = ((bits >> 52) & 0x7FF) - 1023;
    uint64_t mantissa = bits & 0xFFFFFFFFFFFFF;
    // If denormal (exp == -1023), this logic is wrong.
    // But assuming normalized numbers:
    return exp + (mantissa != 0);
}

static inline int8_t iceil_log2(float x) {
    // s1, m24, e7
    uint32_t bits = std::bit_cast<uint32_t>(x);
    uint8_t exp = static_cast<uint8_t>((bits >> 23) & 0xFF);
    uint32_t mant = bits & 0x7FFFFF;
    return static_cast<int8_t>(exp - 127 + (mant != 0));
}

void test(float x) {
    float std_res = std::ceil(std::log2(x));
    float fast_res = iceil_log2((float)x);
    std::cout << "x=" << x << " std=" << std_res << " iceil=" << fast_res;
    if (std_res != fast_res)
        std::cout << " MISMATCH";
    std::cout << "\n";
}

void test_pow2(float x) {
    float std_res = std::log2(x);
    float fast_res = get_exponent_pow2(x);
    std::cout << "POW2 x=" << x << " std=" << std_res << " fast=" << fast_res;
    if (std_res != fast_res)
        std::cout << " MISMATCH";
    std::cout << "\n";
}

int main() {
    std::cout << "Testing Powers of 2 (f calculation)\n";
    test_pow2(1.0);
    test_pow2(2.0);
    test_pow2(0.5);
    test_pow2(0.25);
    test_pow2(1024.0);
    test_pow2(std::pow(2, 20));

    std::cout << "\nTesting General Ceil Log2 (i_high/low)\n";
    test(1.0);
    test(1.1);
    test(2.0);
    test(2.9);
    test(3.0);
    test(4.0);
    test(100.0);
    test(0.5); // log2(0.5) = -1. ceil(-1) = -1. My logic: exp=-1, m=0 -> -1.
    test(0.6); // log2(0.6) = -0.73. ceil = 0. My logic: exp=-1, m!=0 -> 0.
    test(
        0.3
    ); // log2(0.3) = -1.73. ceil = -1. My logic: 0.3 = 1.2*2^-2. exp=-2. m!=0 -> -1.

    // Check 0 handling?
    // fast_ceil_log2(0) behavior needs to match or be safe.
    // std::log2(0) is -inf. ceil(-inf) is -inf.

    return 0;
}
