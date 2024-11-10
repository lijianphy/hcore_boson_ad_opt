/**
 * @file bits.h
 * @brief This header file provides a collection of inline functions for
 *        bit manipulation operations on 64-bit integers.
 *
 * This is a header only library.
 */

#ifndef BITS_H_
#define BITS_H_

#include <stdint.h>

// Check if the compiler supports builtin bit functions
#if defined __GNUC__ || defined __clang__
#define HAVE_BUILTIN_POPCOUNT 1
#define HAVE_BUILTIN_PARITY 1
#define HAVE_BUILTIN_CTZ 1
#define HAVE_BUILTIN_CLZ 1

#if __SIZEOF_LONG__ == 8
#define BUILTIN_POPCOUNT(x) __builtin_popcountl(x)
#define BUILTIN_PARITY(x) __builtin_parityl(x)
#define BUILTIN_CTZ(x) __builtin_ctzl(x)
#define BUILTIN_CLZ(x) __builtin_clzl(x)

#elif __SIZEOF_LONG_LONG__ == 8
#define BUILTIN_POPCOUNT(x) __builtin_popcountll(x)
#define BUILTIN_PARITY(x) __builtin_parityll(x)
#define BUILTIN_CTZ(x) __builtin_ctzll(x)
#define BUILTIN_CLZ(x) __builtin_clzll(x)
#else
#error "uint64_t matches neither long nor long long size"
#endif
#else
#define HAVE_BUILTIN_POPCOUNT 0
#define HAVE_BUILTIN_PARITY 0
#define HAVE_BUILTIN_CTZ 0
#define HAVE_BUILTIN_CLZ 0
#endif

/**
 * Set the nth bit of x to 1, n is 0-indexed
 */
static inline uint64_t set_bit(uint64_t x, int n)
{
    return x | (UINT64_C(1) << n);
}

/**
 * Set the nth bit of x to 0, n is 0-indexed
 */
static inline uint64_t clear_bit(uint64_t x, int n)
{
    return x & ~(UINT64_C(1) << n);
}

/**
 * Toggle (change) the nth bit of x, n is 0-indexed
 */
static inline uint64_t toggle_bit(uint64_t x, int n)
{
    return x ^ (UINT64_C(1) << n);
}

/**
 * Check if the nth bit of x is set, n is 0-indexed
 */
static inline int is_bit_set(uint64_t x, int n)
{
    return (x & (UINT64_C(1) << n)) != 0;
}

/**
 * Check if the nth bit of x is clear, n is 0-indexed
 */
static inline int is_bit_clear(uint64_t x, int n)
{
    return (x & (UINT64_C(1) << n)) == 0;
}

/**
 * Count the number of set bits in a 64-bit integer.
 */
static inline int pop_count(uint64_t x)
{
#if HAVE_BUILTIN_POPCOUNT
    return BUILTIN_POPCOUNT(x);
#else
    x -= ((x >> 1) & UINT64_C(0x5555555555555555));                                     // 0-2 in 2 bits
    x = ((x >> 2) & UINT64_C(0x3333333333333333)) + (x & UINT64_C(0x3333333333333333)); // 0-4 in 4 bits
    x = ((x >> 4) + x) & UINT64_C(0x0f0f0f0f0f0f0f0f);                                  // 0-8 in 8 bits
    x *= UINT64_C(0x0101010101010101);
    return x >> 56;
#endif
}

/**
 * Count the number of bit blocks.
 * A bit block is a sequence of set bits.
 * Example:
 * ..1..11111...111.  -> 3
 * ...1..11111...111  -> 3
 * ......1.....1.1..  -> 3
 * .........111.1111  -> 2
 */
static inline int bit_block_count(uint64_t x)
{
    return (x & 1) + pop_count((x ^ (x >> 1))) / 2;
}

/**
 * Reverse the bits of x
 */
static inline uint64_t reverse_bits(uint64_t x)
{
    x = ((x & UINT64_C(0xaaaaaaaaaaaaaaaa)) >> 1) | ((x & UINT64_C(0x5555555555555555)) << 1);
    x = ((x & UINT64_C(0xcccccccccccccccc)) >> 2) | ((x & UINT64_C(0x3333333333333333)) << 2);
    x = ((x & UINT64_C(0xf0f0f0f0f0f0f0f0)) >> 4) | ((x & UINT64_C(0x0f0f0f0f0f0f0f0f)) << 4);
    x = ((x & UINT64_C(0xff00ff00ff00ff00)) >> 8) | ((x & UINT64_C(0x00ff00ff00ff00ff)) << 8);
    x = ((x & UINT64_C(0xffff0000ffff0000)) >> 16) | ((x & UINT64_C(0x0000ffff0000ffff)) << 16);
    x = (x >> 32) | (x << 32);
    return x;
}

/**
 * Return the parity of x.
 * Parity is 1 if the number of set bits in x is odd
 */
static inline int parity(uint64_t x)
{
#if HAVE_BUILTIN_PARITY
    return BUILTIN_PARITY(x);
#else
    x ^= x >> 1;
    x ^= x >> 2;
    x = (x & UINT64_C(0x1111111111111111)) * UINT64_C(0x1111111111111111);
    return (x >> 60) & 1;
#endif
}

/**
 * Check if x is a power of 2
 */
static inline int is_power_of_2(uint64_t x)
{
    // x is a power of 2 if x & (x - 1) is 0 and x is not 0
    return x && !(x & (x - 1));
}

/**
 * return the position of the rightmost set bit (0-indexed)
 * x should be non-zero
 * the value is same as the number of trailing zeros in x
 */
static inline int rightmost_set_bit(uint64_t x)
{
#if HAVE_BUILTIN_CTZ
    // BUILTIN_CTZ returns the number of trailing zeros in x
    return BUILTIN_CTZ(x);
#else
    static const int table[64] = {
        0, 1, 59, 2, 60, 48, 54, 3,
        61, 40, 49, 28, 55, 34, 43, 4,
        62, 52, 38, 41, 50, 19, 29, 21,
        56, 31, 35, 12, 44, 15, 23, 5,
        63, 58, 47, 53, 39, 27, 33, 42,
        51, 37, 18, 20, 30, 11, 14, 22,
        57, 46, 26, 32, 36, 17, 10, 13,
        45, 25, 16, 9, 24, 8, 7, 6};

    // clear all bits except the rightmost set bit
    x &= ~(x - 1);
    // the magic number is a De Bruijn sequence,
    // see https://en.wikipedia.org/wiki/De_Bruijn_sequence
    return table[(x * UINT64_C(0x03f6eaf2cd271461)) >> 58];
#endif
}

/**
 * Return the position of the leftmost set bit (0-indexed)
 * x should be non-zero
 */
static inline int leftmost_set_bit(uint64_t x)
{
#if HAVE_BUILTIN_CLZ
    // BUILTIN_CLZ returns the number of leading zeros in x
    return 63 - BUILTIN_CLZ(x);
#else
    static const int table[64] = {
        0, 1, 59, 2, 60, 48, 54, 3,
        61, 40, 49, 28, 55, 34, 43, 4,
        62, 52, 38, 41, 50, 19, 29, 21,
        56, 31, 35, 12, 44, 15, 23, 5,
        63, 58, 47, 53, 39, 27, 33, 42,
        51, 37, 18, 20, 30, 11, 14, 22,
        57, 46, 26, 32, 36, 17, 10, 13,
        45, 25, 16, 9, 24, 8, 7, 6};

    // set all bits to the right of the leftmost set bit
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    // clear all bits except the leftmost set bit
    x = (x >> 1) + 1;
    // the magic number is a De Bruijn sequence,
    // see https://en.wikipedia.org/wiki/De_Bruijn_sequence
    return table[(x * UINT64_C(0x03f6eaf2cd271461)) >> 58];
#endif
}

/**
 * Return the Gray code of x
 * Gray code is a binary numeral system where two successive values differ in only one bit
 * See https://en.wikipedia.org/wiki/Gray_code
 */
static inline uint64_t gray_code(uint64_t x)
{
    return x ^ (x >> 1);
}

/**
 * Return the binary code of the Gray code x
 */
static inline uint64_t inverse_gray_code(uint64_t x)
{
    x ^= x >> 1;
    x ^= x >> 2;
    x ^= x >> 4;
    x ^= x >> 8;
    x ^= x >> 16;
    x ^= x >> 32;
    return x;
}

/**
 * Rotate x left by n bits
 */
static inline uint64_t rotate_left(uint64_t x, int n)
{
    return (x << n) | (x >> (64 - n));
}

/**
 * Rotate x right by n bits
 */
static inline uint64_t rotate_right(uint64_t x, int n)
{
    return (x >> n) | (x << (64 - n));
}

/**
 * Swap the mth and nth bits of x
 */
static inline uint64_t swap_bits(uint64_t x, int m, int n)
{
    uint64_t y = ((x >> m) ^ (x >> n)) & 1; // one if bits differ
    x ^= (y << m);                          // change if bits differ
    x ^= (y << n);                          // change if bits differ
    return x;
}

/**
 * Print the bits of a 64-bit integer
 * n is the number of bits to print
 */
static inline void print_bits(uint64_t v, int n)
{
    for (int i = n - 1; i >= 0; i--)
    {
        printf("%d", (int)((v >> i) & 1));
    }
    printf("\n");
}

#endif // BITS_H_
