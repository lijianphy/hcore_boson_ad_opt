/**
 * @file combination.c
 * @brief Utility functions for combinatorial calculations
 */

#include <stdint.h>
#include <stdio.h>
#include "combination.h"
#include "bits.h"
#include "bits128.h"

/**
 * Calculate the greatest common divisor of a and b
 */
uint64_t gcd(uint64_t a, uint64_t b)
{
    while (b != 0)
    {
        uint64_t r;
        r = a % b;
        a = b;
        b = r;
    }
    return a;
}

/**
 * Calculate the factorial of n
 * n should <= 20
 * For n<=20, n! fits into 64 bits.
 */
uint64_t factorial(int n)
{
    static const uint64_t result[21] = {
        1, 1, 2, 6, 24, 120, 720, 5040, 40320,
        362880, 3628800, 39916800, 479001600,
        6227020800, 87178291200, 1307674368000,
        20922789888000, 355687428096000,
        6402373705728000, 121645100408832000,
        2432902008176640000};
    return result[n];
}

/**
 * Calculate the binomial coefficient "n choose k"
 * n and k should be non-negative
 */
uint64_t binomial(int n, int k)
{
    // some special cases
    if (k == 0 || k == n) return 1;
    if (k > n) return 0;
    if (n <= 20) return factorial(n)/(factorial(k) * factorial(n - k));

    if (k * 2 > n)
    {
        k = n - k;
    }

    uint64_t ret = n - k + 1;
    int i = 2;
    int j = ret + 1;
    while (i <= k)
    {
        ret *= j;
        ret /= i;
        i++;
        j++;
    }

    return ret;
}

/**
 * Compute the lexicographically next bit permutation
 * v is the current permutation
 * Ref: https://graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation
 */
uint64_t next_bit_permutation(uint64_t v)
{
    uint64_t t = v | (v - 1); // t gets v's least significant 0 bits set to 1
    // Next set to 1 the most significant bit to change,
    uint64_t t1 = t + 1;
    // set to 0 the least significant ones, and add the necessary 1 bits.
    return t1 | (((~t & t1) - 1) >> (rightmost_set_bit(v) + 1));
}

uint128_t next_bit_permutation128(uint128_t v)
{
    uint128_t t = v | (v - 1); // t gets v's least significant 0 bits set to 1
    // Next set to 1 the most significant bit to change,
    uint128_t t1 = t + 1;
    // set to 0 the least significant ones, and add the necessary 1 bits.
    return t1 | (((~t & t1) - 1) >> (rightmost_set_bit128(v) + 1));
}

/**
 * Compute the lexicographically index of bit permutation
 * k is the number of set bits in data
 * data is the bit permutation
 */
uint64_t permutation2index(int k, uint64_t data)
{
    uint64_t t = data & (data + 1);
    if (t == 0) return 0;

    int n1 = leftmost_set_bit(data);
    int n2 = rightmost_set_bit(t);
    uint64_t index = 0;
    for (int i = n1, j = k; i >= n2; i--)
    {
        if (data & (UINT64_C(1) << i))
        {
            index += binomial(i, j);
            j--;
        }
    }
    return index;
}


uint64_t permutation2index128(int k, uint128_t data)
{
    uint128_t t = data & (data + 1);
    if (t == 0) return 0;

    int n1 = leftmost_set_bit128(data);
    int n2 = rightmost_set_bit128(t);
    uint64_t index = 0;
    for (int i = n1, j = k; i >= n2; i--)
    {
        if (data & (UINT128_C(1) << i))
        {
            index += binomial(i, j);
            j--;
        }
    }
    return index;
}


/**
 * Compute the lexicographically bit permutation from index
 * n is the total width of the bit permutation
 * k is the number of set bits
 */
uint64_t index2permutation(int n, int k, uint64_t index)
{
    // some special cases
    if (k == 0) return 0;
    if (index == 0) return (UINT64_C(1) << k) - 1;

    uint64_t data = 0;
    int j = k;
    for (int i = n-1; i >= 0; i--)
    {
        uint64_t temp = binomial(i, j);
        if (index >= temp)
        {
            data |= (UINT64_C(1) << i);
            index -= temp;
            j--;

            if (j == 0) break;
            if (index == 0) {
                data |= (UINT64_C(1) << j) - 1;
                break;
            }
        }
    }
    return data;
}

uint128_t index2permutation128(int n, int k, uint64_t index)
{
    // some special cases
    if (k == 0) return 0;
    if (index == 0) return (UINT128_C(1) << k) - 1;

    uint128_t data = 0;
    int j = k;
    for (int i = n-1; i >= 0; i--)
    {
        uint64_t temp = binomial(i, j);
        if (index >= temp)
        {
            data |= (UINT128_C(1) << i);
            index -= temp;
            j--;

            if (j == 0) break;
            if (index == 0) {
                data |= (UINT128_C(1) << j) - 1;
                break;
            }
        }
    }
    return data;
}
