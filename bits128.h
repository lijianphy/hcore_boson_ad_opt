/**
 * @file bits128.h
 * @brief This header file provides a collection of inline functions for
 *       bit manipulation operations on 128-bit integers.
 */

#ifndef BITS128_H_
#define BITS128_H_

#include <stdint.h>
#include <stdio.h>
#include "bits.h"

// // Check if the compiler supports 128-bit integers
// #if defined(__SIZEOF_INT128__) || defined(__SIZEOF_INT128)
// #define HAVE_INT128 1
// #else
// #define HAVE_INT128 0
// #endif

typedef __uint128_t uint128_t;

#define UINT128_C(c) (uint128_t)(c##ULL) // define a 128-bit literal
#define UINT128_C2(hi, lo) ((uint128_t)(hi) << 64 | (uint128_t)(lo))

/**
 * Set the nth bit of x to 1, n is 0-indexed
 */
static inline uint128_t set_bit128(uint128_t x, int n)
{
    return x | (UINT128_C(1) << n);
}

/**
 * Set the nth bit of x to 0, n is 0-indexed
 */
static inline uint128_t clear_bit128(uint128_t x, int n)
{
    return x & ~(UINT128_C(1) << n);
}

/**
 * Toggle (change) the nth bit of x, n is 0-indexed
 */
static inline uint128_t toggle_bit128(uint128_t x, int n)
{
    return x ^ (UINT128_C(1) << n);
}

/**
 * Check if the nth bit of x is set, n is 0-indexed
 */
static inline int is_bit_set128(uint128_t x, int n)
{
    return (x & (UINT128_C(1) << n)) != 0;
}

/**
 * Check if the nth bit of x is clear, n is 0-indexed
 */
static inline int is_bit_clear128(uint128_t x, int n)
{
    return (x & (UINT128_C(1) << n)) == 0;
}

/**
 * Count the number of set bits in a 128-bit integer.
 */
static inline int pop_count128(uint128_t x)
{
    return pop_count((uint64_t)(x >> 64)) + pop_count((uint64_t)x);
}

/**
 * Count the number of bit blocks in a 128-bit integer.
 */
static inline int bit_block_count128(uint128_t x)
{
    return (x & 1) + pop_count128((x ^ (x >> 1))) / 2;
}

/**
 * Reverse the bits of x
 */
static inline uint128_t reverse_bits128(uint128_t x)
{
    uint64_t lo = (uint64_t)x;
    uint64_t hi = (uint64_t)(x >> 64);
    return ((uint128_t)reverse_bits(lo) << 64) | reverse_bits(hi);
}

/**
 * Return the parity of x
 */
static inline int parity128(uint128_t x)
{
    return parity((uint64_t)(x >> 64)) ^ parity((uint64_t)x);
}

/**
 * Check if x is a power of 2
 */
static inline int is_power_of_2_128(uint128_t x)
{
    return x && !(x & (x - 1));
}

/**
 * Return the position of the rightmost set bit (0-indexed)
 * x should be non-zero
 */
static inline int rightmost_set_bit128(uint128_t x)
{
    uint64_t lo = (uint64_t)x;
    if (lo != 0)
        return rightmost_set_bit(lo);
    uint64_t hi = (uint64_t)(x >> 64);
    return hi != 0 ? rightmost_set_bit(hi) + 64 : -1;
}

/**
 * Return the position of the leftmost set bit (0-indexed)
 * x should be non-zero
 */
static inline int leftmost_set_bit128(uint128_t x)
{
    uint64_t hi = (uint64_t)(x >> 64);
    if (hi != 0)
        return leftmost_set_bit(hi) + 64;
    uint64_t lo = (uint64_t)x;
    return lo != 0 ? leftmost_set_bit(lo) : -1;
}

/**
 * Return the Gray code of x
 */
static inline uint128_t gray_code128(uint128_t x)
{
    return x ^ (x >> 1);
}

/**
 * Return the binary code of the Gray code x
 */
static inline uint128_t inverse_gray_code128(uint128_t x)
{
    x ^= x >> 1;
    x ^= x >> 2;
    x ^= x >> 4;
    x ^= x >> 8;
    x ^= x >> 16;
    x ^= x >> 32;
    x ^= x >> 64;
    return x;
}

/**
 * Rotate x left by n bits
 */
static inline uint128_t rotate_left128(uint128_t x, int n)
{
    return (x << n) | (x >> (128 - n));
}

/**
 * Rotate x right by n bits
 */
static inline uint128_t rotate_right128(uint128_t x, int n)
{
    return (x >> n) | (x << (128 - n));
}

/**
 * Swap the mth and nth bits of x
 */
static inline uint128_t swap_bits128(uint128_t x, int m, int n)
{
    uint128_t y = ((x >> m) ^ (x >> n)) & 1;
    x ^= (y << m);
    x ^= (y << n);
    return x;
}

/**
 * Print the bits of a 128-bit integer
 */
static inline void print_bits128(uint128_t v, int n)
{
    for (int i = n - 1; i >= 0; i--)
    {
        printf("%d", (int)((v >> i) & 1));
    }
    printf("\n");
}

#endif // BITS128_H_
