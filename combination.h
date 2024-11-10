/**
 * @file combination.h
 * @brief Utility functions for combinatorial calculations
 */

#ifndef COMBINATION_H_
#define COMBINATION_H_

#include <stdint.h>
#include <stdio.h>

/**
 * Calculate the greatest common divisor of a and b
 */
uint64_t gcd(uint64_t a, uint64_t b);

/**
 * Calculate the factorial of n
 * n should <= 20
 * For n<=20, n! fits into 64 bits.
 */
uint64_t factorial(int n);

/**
 * Calculate the binomial coefficient "n choose k"
 * n and k should be non-negative
 */
uint64_t binomial(int n, int k);

/**
 * Compute the lexicographically next bit permutation
 * v is the current permutation
 * Ref: https://graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation
 */
uint64_t next_bit_permutation(uint64_t v);

/**
 * Compute the lexicographically index of bit permutation (0-indexed)
 * k is the number of set bits in data
 * data is the bit permutation
 */
uint64_t permutation2index(int k, uint64_t data);

/**
 * Compute the lexicographically bit permutation from index
 * n is the total width of the bit permutation
 * k is the number of set bits
 */
uint64_t index2permutation(int n, int k, uint64_t index);

#endif // COMBINATION_H_
