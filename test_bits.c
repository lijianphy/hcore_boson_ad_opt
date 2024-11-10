/**
 * @file test_bits.c
 * @brief Unit tests for bit manipulation functions
 */

#include <stdio.h> // for printf()
#include <stdlib.h>  // for rand()
#include <CUnit/CUnit.h>
#include <CUnit/Basic.h>
#include "bits.h"


static int pop_count_naive(uint64_t x) {
    int count = 0;
    while (x) {
        count += x & 1;
        x >>= 1;
    }
    return count;
}

static int bit_block_count_naive(uint64_t x) {
    int count = 0;
    int in_block = 0;

    while (x) {
        if (x & 1) {
            if (!in_block) {
                count++;
                in_block = 1;
            }
        } else {
            in_block = 0;
        }
        x >>= 1;
    }
    return count;
}


static uint64_t reverse_bits_naive(uint64_t x) {
    uint64_t y = 0;
    for (int i = 0; i < 64; i++) {
        y <<= 1;
        y |= x & 1;
        x >>= 1;
    }
    return y;
}

static int parity_naive(uint64_t x) {
    int count = pop_count_naive(x);
    return count % 2;
}

static int is_power_of_2_naive(uint64_t x) {
    return pop_count_naive(x) == 1;
}

static int rightmost_set_bit_naive(uint64_t x) {
    if (x == 0) {
        return -1;
    }
    int i = 0;
    while ((x & 1) == 0) {
        x >>= 1;
        i++;
    }
    return i;
}

static int leftmost_set_bit_naive(uint64_t x) {
    if (x == 0) {
        return -1;
    }
    int i = 0;
    while (x >>= 1) {
        i++;
    }
    return i;
}

void test_set_bit() {
    CU_ASSERT_EQUAL(set_bit(0, 0), 1); // 0001
    CU_ASSERT_EQUAL(set_bit(0, 1), 2); // 0010
    CU_ASSERT_EQUAL(set_bit(1, 1), 3); // 0011
    CU_ASSERT_EQUAL(set_bit(2, 0), 3); // 0011
    CU_ASSERT_EQUAL(set_bit(3, 0), 3); // 0011
    CU_ASSERT_EQUAL(set_bit(3, 1), 3); // 0011
    CU_ASSERT_EQUAL(set_bit(3, 2), 7); // 0111
    CU_ASSERT_EQUAL(set_bit(3, 3), 11); // 1011
}

void test_clear_bit() {
    CU_ASSERT_EQUAL(clear_bit(1, 0), 0); // 0000
    CU_ASSERT_EQUAL(clear_bit(3, 1), 1); // 0001
    CU_ASSERT_EQUAL(clear_bit(2, 1), 0); // 0000
}

void test_toggle_bit() {
    CU_ASSERT_EQUAL(toggle_bit(0, 0), 1); // 0001
    CU_ASSERT_EQUAL(toggle_bit(1, 0), 0); // 0000
    CU_ASSERT_EQUAL(toggle_bit(2, 1), 0); // 0000
}

void test_is_bit_set() {
    CU_ASSERT_EQUAL(is_bit_set(1, 0), 1); // 0001
    CU_ASSERT_EQUAL(is_bit_set(2, 1), 1); // 0010
    CU_ASSERT_EQUAL(is_bit_set(0, 0), 0); // 0000
}

void test_is_bit_clear() {
    CU_ASSERT_EQUAL(is_bit_clear(1, 1), 1);
    CU_ASSERT_EQUAL(is_bit_clear(2, 0), 1);
    CU_ASSERT_EQUAL(is_bit_clear(0, 0), 1);
}

void test_pop_count() {
    CU_ASSERT_EQUAL(pop_count(0), 0);
    CU_ASSERT_EQUAL(pop_count(1), 1);
    CU_ASSERT_EQUAL(pop_count(3), 2);
    CU_ASSERT_EQUAL(pop_count(0x15), 3); // 10101
    CU_ASSERT_EQUAL(pop_count(0x1C), 3); // 11100
    CU_ASSERT_EQUAL(pop_count(0x1B), 4); // 11011
    for (uint64_t x = 0; x < 1000; x++) {
        uint64_t y = 1234567 + x;
        uint64_t z = 5678901 + x;
        uint64_t w = rand();
        CU_ASSERT_EQUAL(pop_count(x), pop_count_naive(x));
        CU_ASSERT_EQUAL(pop_count(y), pop_count_naive(y));
        CU_ASSERT_EQUAL(pop_count(z), pop_count_naive(z));
        CU_ASSERT_EQUAL(pop_count(w), pop_count_naive(w));
    }
}

void test_bit_block_count() {
    CU_ASSERT_EQUAL(bit_block_count(0x15), 3); // 10101
    CU_ASSERT_EQUAL(bit_block_count(0x1C), 1); // 11100
    CU_ASSERT_EQUAL(bit_block_count(0x1B), 2); // 11011
    for (uint64_t x = 0; x < 1000; x++) {
        uint64_t y = 1234567 + x;
        uint64_t z = 5678901 + x;
        uint64_t w = rand();
        CU_ASSERT_EQUAL(bit_block_count(x), bit_block_count_naive(x));
        CU_ASSERT_EQUAL(bit_block_count(y), bit_block_count_naive(y));
        CU_ASSERT_EQUAL(bit_block_count(z), bit_block_count_naive(z));
        CU_ASSERT_EQUAL(bit_block_count(w), bit_block_count_naive(w));
    }
}

void test_reverse_bits() {
    CU_ASSERT_EQUAL(reverse_bits(0x1), UINT64_C(0x8000000000000000));
    CU_ASSERT_EQUAL(reverse_bits(0x2), UINT64_C(0x4000000000000000));
    CU_ASSERT_EQUAL(reverse_bits(0x3), UINT64_C(0xC000000000000000));
    for (uint64_t x = 0; x < 1000; x++) {
        uint64_t y = 1234567 + x;
        uint64_t z = 5678901 + x;
        uint64_t w = rand();
        CU_ASSERT_EQUAL(reverse_bits(x), reverse_bits_naive(x));
        CU_ASSERT_EQUAL(reverse_bits(y), reverse_bits_naive(y));
        CU_ASSERT_EQUAL(reverse_bits(z), reverse_bits_naive(z));
        CU_ASSERT_EQUAL(reverse_bits(w), reverse_bits_naive(w));
    }
}

void test_parity() {
    CU_ASSERT_EQUAL(parity(0x1), 1);
    CU_ASSERT_EQUAL(parity(0x3), 0);
    CU_ASSERT_EQUAL(parity(0x5), 0);
    for (uint64_t x = 0; x < 1000; x++) {
        uint64_t y = 1234567 + x;
        uint64_t z = 5678901 + x;
        uint64_t w = rand();
        CU_ASSERT_EQUAL(parity(x), parity_naive(x));
        CU_ASSERT_EQUAL(parity(y), parity_naive(y));
        CU_ASSERT_EQUAL(parity(z), parity_naive(z));
        CU_ASSERT_EQUAL(parity(w), parity_naive(w));
    }
}

void test_is_power_of_2() {
    CU_ASSERT_EQUAL(is_power_of_2(1), 1);
    CU_ASSERT_EQUAL(is_power_of_2(2), 1);
    CU_ASSERT_EQUAL(is_power_of_2(3), 0);
    CU_ASSERT_EQUAL(is_power_of_2(4), 1);
    CU_ASSERT_EQUAL(is_power_of_2(5), 0);
    CU_ASSERT_EQUAL(is_power_of_2(6), 0);
    for (uint64_t x = 0; x < 1000; x++) {
        uint64_t y = 1234567 + x;
        uint64_t z = 5678901 + x;
        uint64_t w = rand();
        CU_ASSERT_EQUAL(is_power_of_2(x), is_power_of_2_naive(x));
        CU_ASSERT_EQUAL(is_power_of_2(y), is_power_of_2_naive(y));
        CU_ASSERT_EQUAL(is_power_of_2(z), is_power_of_2_naive(z));
        CU_ASSERT_EQUAL(is_power_of_2(w), is_power_of_2_naive(w));
    }
}

void test_rightmost_set_bit() {
    CU_ASSERT_EQUAL(rightmost_set_bit(0x1), 0);
    CU_ASSERT_EQUAL(rightmost_set_bit(0x2), 1);
    CU_ASSERT_EQUAL(rightmost_set_bit(0x4), 2);
    CU_ASSERT_EQUAL(rightmost_set_bit(UINT64_C(0x8000000000000000)), 63);
    for (uint64_t x = 1; x < 1000; x++) {
        uint64_t y = 1234567 + x;
        uint64_t z = 5678901 + x;
        uint64_t w = rand() + x;
        CU_ASSERT_EQUAL(rightmost_set_bit(x), rightmost_set_bit_naive(x));
        CU_ASSERT_EQUAL(rightmost_set_bit(y), rightmost_set_bit_naive(y));
        CU_ASSERT_EQUAL(rightmost_set_bit(z), rightmost_set_bit_naive(z));
        CU_ASSERT_EQUAL(rightmost_set_bit(w), rightmost_set_bit_naive(w));
    }
}

void test_leftmost_set_bit() {
    CU_ASSERT_EQUAL(leftmost_set_bit(0x1), 0);
    CU_ASSERT_EQUAL(leftmost_set_bit(0x2), 1);
    CU_ASSERT_EQUAL(leftmost_set_bit(0x4), 2);
    for (uint64_t x = 1; x < 1000; x++) {
        uint64_t y = 1234567 + x;
        uint64_t z = 5678901 + x;
        uint64_t w = rand() + 1;
        CU_ASSERT_EQUAL(leftmost_set_bit(x), leftmost_set_bit_naive(x));
        CU_ASSERT_EQUAL(leftmost_set_bit(y), leftmost_set_bit_naive(y));
        CU_ASSERT_EQUAL(leftmost_set_bit(z), leftmost_set_bit_naive(z));
        CU_ASSERT_EQUAL(leftmost_set_bit(w), leftmost_set_bit_naive(w));
    }
}

void test_gray_code() {
    CU_ASSERT_EQUAL(gray_code(0), 0);
    CU_ASSERT_EQUAL(gray_code(1), 1);
    CU_ASSERT_EQUAL(gray_code(2), 3);
}

void test_inverse_gray_code() {
    CU_ASSERT_EQUAL(inverse_gray_code(0), 0);
    CU_ASSERT_EQUAL(inverse_gray_code(1), 1);
    CU_ASSERT_EQUAL(inverse_gray_code(3), 2);
    for (uint64_t x = 0; x < 1000; x++) {
        uint64_t w = rand();
        CU_ASSERT_EQUAL(inverse_gray_code(gray_code(x)), x);
        CU_ASSERT_EQUAL(inverse_gray_code(gray_code(w)), w);
    }
}

void test_rotate_left() {
    CU_ASSERT_EQUAL(rotate_left(1, 1), 2);
    CU_ASSERT_EQUAL(rotate_left(1, 2), 4);
    CU_ASSERT_EQUAL(rotate_left(1, 63), UINT64_C(0x8000000000000000));
}

void test_rotate_right() {
    CU_ASSERT_EQUAL(rotate_right(1, 1), UINT64_C(0x8000000000000000));
    CU_ASSERT_EQUAL(rotate_right(2, 1), 1);
    CU_ASSERT_EQUAL(rotate_right(4, 2), 1);
}

void test_swap_bits() {
    CU_ASSERT_EQUAL(swap_bits(0x5, 0, 2), 0x5);
    CU_ASSERT_EQUAL(swap_bits(0x5, 0, 1), 0x6);
    CU_ASSERT_EQUAL(swap_bits(0x6, 1, 2), 0x6);
}

int main() {
    CU_initialize_registry();

    CU_pSuite suite = CU_add_suite("bit_operations_suite", 0, 0);

    CU_add_test(suite, "test_set_bit", test_set_bit);
    CU_add_test(suite, "test_clear_bit", test_clear_bit);
    CU_add_test(suite, "test_toggle_bit", test_toggle_bit);
    CU_add_test(suite, "test_is_bit_set", test_is_bit_set);
    CU_add_test(suite, "test_is_bit_clear", test_is_bit_clear);
    CU_add_test(suite, "test_pop_count", test_pop_count);
    CU_add_test(suite, "test_bit_block_count", test_bit_block_count);
    CU_add_test(suite, "test_reverse_bits", test_reverse_bits);
    CU_add_test(suite, "test_parity", test_parity);
    CU_add_test(suite, "test_is_power_of_2", test_is_power_of_2);
    CU_add_test(suite, "test_rightmost_set_bit", test_rightmost_set_bit);
    CU_add_test(suite, "test_leftmost_set_bit", test_leftmost_set_bit);
    CU_add_test(suite, "test_gray_code", test_gray_code);
    CU_add_test(suite, "test_inverse_gray_code", test_inverse_gray_code);
    CU_add_test(suite, "test_rotate_left", test_rotate_left);
    CU_add_test(suite, "test_rotate_right", test_rotate_right);
    CU_add_test(suite, "test_swap_bits", test_swap_bits);

    CU_basic_set_mode(CU_BRM_VERBOSE);
    CU_basic_run_tests();
    CU_cleanup_registry();

    return 0;
}
