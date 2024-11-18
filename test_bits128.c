/**
 * @file test_bits128.c
 * @brief Unit tests for 128-bit manipulation functions
 */

#include <stdio.h>
#include <stdlib.h>
#include <CUnit/CUnit.h>
#include <CUnit/Basic.h>
#include "bits128.h"

static int pop_count_naive128(uint128_t x)
{
    int count = 0;
    while (x)
    {
        count += x & 1;
        x >>= 1;
    }
    return count;
}

static int bit_block_count_naive128(uint128_t x)
{
    int count = 0;
    int in_block = 0;

    while (x)
    {
        if (x & 1)
        {
            if (!in_block)
            {
                count++;
                in_block = 1;
            }
        }
        else
        {
            in_block = 0;
        }
        x >>= 1;
    }
    return count;
}

static uint128_t reverse_bits_naive128(uint128_t x)
{
    uint128_t y = 0;
    for (int i = 0; i < 128; i++)
    {
        y <<= 1;
        y |= x & 1;
        x >>= 1;
    }
    return y;
}

static int parity_naive128(uint128_t x)
{
    int count = pop_count_naive128(x);
    return count % 2;
}

static int is_power_of_2_naive128(uint128_t x)
{
    return pop_count_naive128(x) == 1;
}

static int rightmost_set_bit_naive128(uint128_t x)
{
    if (x == 0)
    {
        return -1;
    }
    int i = 0;
    while ((x & 1) == 0)
    {
        x >>= 1;
        i++;
    }
    return i;
}

static int leftmost_set_bit_naive128(uint128_t x)
{
    if (x == 0)
    {
        return -1;
    }
    int i = 0;
    while (x >>= 1)
    {
        i++;
    }
    return i;
}

static uint128_t swap_bits128_naive(uint128_t x, int m, int n)
{
    // Get values of bits at positions m and n
    uint128_t bit_m = (x >> m) & 1;
    uint128_t bit_n = (x >> n) & 1;

    // If bits are different, flip them
    if (bit_m != bit_n)
    {
        // Create a mask with 1s at positions m and n
        uint128_t mask = (UINT128_C(1) << m) | (UINT128_C(1) << n);
        // XOR with mask to flip both bits
        x ^= mask;
    }
    return x;
}

void test_set_bit128()
{
    CU_ASSERT_EQUAL(set_bit128(0, 0), UINT128_C(1));
    CU_ASSERT_EQUAL(set_bit128(0, 1), UINT128_C(2));
    CU_ASSERT_EQUAL(set_bit128(0, 127), UINT128_C2(0x8000000000000000, 0));
    CU_ASSERT_EQUAL(set_bit128(0, 64), UINT128_C2(1, 0));
    CU_ASSERT_EQUAL(set_bit128(0, 65), UINT128_C2(2, 0));
    CU_ASSERT_EQUAL(set_bit128(UINT128_C(1), 0), UINT128_C(1));
    for (uint64_t x = 0; x < 1000; x++)
    {
        uint64_t y = 1234567 + x;
        uint64_t z = 5678901 + x;
        uint64_t w = rand() + 1;
        CU_ASSERT_EQUAL(set_bit128(x, 0), set_bit(x, 0));
        CU_ASSERT_EQUAL(set_bit128(y, 1), set_bit(y, 1));
        CU_ASSERT_EQUAL(set_bit128(z, 2), set_bit(z, 2));
        CU_ASSERT_EQUAL(set_bit128(w, 3), set_bit(w, 3));
    }
}

void test_clear_bit128()
{
    CU_ASSERT_EQUAL(clear_bit128(UINT128_C(1), 0), 0);
    CU_ASSERT_EQUAL(clear_bit128(UINT128_C(3), 0), 2);
    CU_ASSERT_EQUAL(clear_bit128(UINT128_C(3), 1), 1);
    CU_ASSERT_EQUAL(clear_bit128(UINT128_C2(0x8000000000000000, 0), 127), 0);
    for (uint64_t x = 0; x < 1000; x++)
    {
        uint64_t y = 1234567 + x;
        uint64_t z = 5678901 + x;
        uint64_t w = rand() + 1;
        CU_ASSERT_EQUAL(clear_bit128(x, 0), clear_bit(x, 0));
        CU_ASSERT_EQUAL(clear_bit128(y, 1), clear_bit(y, 1));
        CU_ASSERT_EQUAL(clear_bit128(z, 2), clear_bit(z, 2));
        CU_ASSERT_EQUAL(clear_bit128(w, 3), clear_bit(w, 3));
    }
}

void test_toggle_bit128()
{
    CU_ASSERT_EQUAL(toggle_bit128(0, 0), UINT128_C(1));
    CU_ASSERT_EQUAL(toggle_bit128(UINT128_C(1), 0), 0);
    CU_ASSERT_EQUAL(toggle_bit128(0, 127), UINT128_C2(0x8000000000000000, 0));
    for (uint64_t x = 0; x < 1000; x++)
    {
        uint64_t y = 1234567 + x;
        uint64_t z = 5678901 + x;
        uint64_t w = rand() + 1;
        CU_ASSERT_EQUAL(toggle_bit128(x, 0), toggle_bit(x, 0));
        CU_ASSERT_EQUAL(toggle_bit128(y, 1), toggle_bit(y, 1));
        CU_ASSERT_EQUAL(toggle_bit128(z, 2), toggle_bit(z, 2));
        CU_ASSERT_EQUAL(toggle_bit128(w, 3), toggle_bit(w, 3));
    }
}

void test_is_bit_set128()
{
    CU_ASSERT_EQUAL(is_bit_set128(UINT128_C(1), 0), 1);
    CU_ASSERT_EQUAL(is_bit_set128(UINT128_C(2), 1), 1);
    CU_ASSERT_EQUAL(is_bit_set128(0, 0), 0);
    CU_ASSERT_EQUAL(is_bit_set128(UINT128_C2(0x8000000000000000, 0), 127), 1);
    for (uint64_t x = 0; x < 1000; x++)
    {
        uint64_t y = 1234567 + x;
        uint64_t z = 5678901 + x;
        uint64_t w = rand() + 1;
        CU_ASSERT_EQUAL(is_bit_set128(x, 0), is_bit_set(x, 0));
        CU_ASSERT_EQUAL(is_bit_set128(y, 1), is_bit_set(y, 1));
        CU_ASSERT_EQUAL(is_bit_set128(z, 2), is_bit_set(z, 2));
        CU_ASSERT_EQUAL(is_bit_set128(w, 3), is_bit_set(w, 3));
    }
}

void test_is_bit_clear128()
{
    CU_ASSERT_EQUAL(is_bit_clear128(UINT128_C(1), 1), 1);
    CU_ASSERT_EQUAL(is_bit_clear128(UINT128_C(2), 0), 1);
    CU_ASSERT_EQUAL(is_bit_clear128(0, 0), 1);
    CU_ASSERT_EQUAL(is_bit_clear128(UINT128_C2(0x8000000000000000, 0), 126), 1);
    for (uint64_t x = 0; x < 1000; x++)
    {
        uint64_t y = 1234567 + x;
        uint64_t z = 5678901 + x;
        uint64_t w = rand() + 1;
        CU_ASSERT_EQUAL(is_bit_clear128(x, 0), is_bit_clear(x, 0));
        CU_ASSERT_EQUAL(is_bit_clear128(y, 1), is_bit_clear(y, 1));
        CU_ASSERT_EQUAL(is_bit_clear128(z, 2), is_bit_clear(z, 2));
        CU_ASSERT_EQUAL(is_bit_clear128(w, 3), is_bit_clear(w, 3));
    }
}

static uint128_t test_values[] = {
    0,
    UINT128_C(1),
    UINT128_C(3),
    UINT128_C2(0x8000000000000000, 0),
    UINT128_C2(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF),
    UINT128_C2(0x123456789ABCDEF0, 0xFEDCBA9876543210),
    UINT128_C2(0x0F0F0F0F0F0F0F0F, 0xF0F0F0F0F0F0F0F0),
    UINT128_C2(0xAAAAAAAAAAAAAAAA, 0x5555555555555555),
    UINT128_C2(0xFFFFFFFF00000000, 0x00000000FFFFFFFF),
    UINT128_C2(0x1234567890ABCDEF, 0xFEDCBA0987654321),
    UINT128_C2(0x0F0E0D0C0B0A0908, 0x0706050403020100),
    UINT128_C2(0x1111111111111111, 0x2222222222222222),
    UINT128_C2(0x3333333333333333, 0x4444444444444444),
    UINT128_C2(0x5555555555555555, 0x6666666666666666),
    UINT128_C2(0x7777777777777777, 0x8888888888888888),
    UINT128_C2(0x9999999999999999, 0xAAAAAAAAAAAAAAAA),
    UINT128_C2(0xBBBBBBBBBBBBBBBB, 0xCCCCCCCCCCCCCCCC),
    UINT128_C2(0xDDDDDDDDDDDDDDDD, 0xEEEEEEEEEEEEEEEE),
    UINT128_C2(0xFFFFFFFFFFFFFFFF, 0x0000000000000000),
    UINT128_C2(0x0000000000000000, 0x1111111111111111),
    UINT128_C2(0x2222222222222222, 0x3333333333333333),
    UINT128_C2(0x4444444444444444, 0x5555555555555555),
    UINT128_C2(0x6666666666666666, 0x7777777777777777),
    UINT128_C2(0x8888888888888888, 0x9999999999999999),
    UINT128_C2(0xAAAAAAAAAAAAAA00, 0xBBBBBBBBBBBBBBBB),
    UINT128_C2(0xCCCCCCCCCCCCCCCC, 0x0000000000000000),
    UINT128_C2(0x0F0F0F0F0F0F0F0F, 0x0000FFFFFFFFFFFF),
    UINT128_C2(0x00000000FFFFFFFF, 0x0F0F0F0F0F0F0F0F),
    UINT128_C2(0x1234567890123456, 0x7890123456789012)};

void test_pop_count128()
{
    for (size_t i = 0; i < sizeof(test_values) / sizeof(test_values[0]); i++)
    {
        CU_ASSERT_EQUAL(pop_count128(test_values[i]), pop_count_naive128(test_values[i]));
    }
    for (uint64_t x = 0; x < 1000; x++)
    {
        uint128_t y = UINT128_C2(1234567, x);
        uint128_t z = UINT128_C2(x, 5678901);
        uint128_t w = UINT128_C2(rand(), x);
        CU_ASSERT_EQUAL(pop_count128(x), pop_count_naive128(x));
        CU_ASSERT_EQUAL(pop_count128(y), pop_count_naive128(y));
        CU_ASSERT_EQUAL(pop_count128(z), pop_count_naive128(z));
        CU_ASSERT_EQUAL(pop_count128(w), pop_count_naive128(w));
    }
}

void test_bit_block_count128()
{
    for (size_t i = 0; i < sizeof(test_values) / sizeof(test_values[0]); i++)
    {
        CU_ASSERT_EQUAL(bit_block_count128(test_values[i]), bit_block_count_naive128(test_values[i]));
    }
    for (uint64_t x = 0; x < 1000; x++)
    {
        uint128_t y = UINT128_C2(1234567, x);
        uint128_t z = UINT128_C2(x, 5678901);
        uint128_t w = UINT128_C2(rand(), x);
        CU_ASSERT_EQUAL(bit_block_count128(x), bit_block_count_naive128(x));
        CU_ASSERT_EQUAL(bit_block_count128(y), bit_block_count_naive128(y));
        CU_ASSERT_EQUAL(bit_block_count128(z), bit_block_count_naive128(z));
        CU_ASSERT_EQUAL(bit_block_count128(w), bit_block_count_naive128(w));
    }
}

void test_reverse_bits128()
{
    for (size_t i = 0; i < sizeof(test_values) / sizeof(test_values[0]); i++)
    {
        CU_ASSERT_EQUAL(reverse_bits128(test_values[i]), reverse_bits_naive128(test_values[i]));
    }
    for (uint64_t x = 0; x < 1000; x++)
    {
        uint128_t y = UINT128_C2(1234567, x);
        uint128_t z = UINT128_C2(x, 5678901);
        uint128_t w = UINT128_C2(rand(), x);
        CU_ASSERT_EQUAL(reverse_bits128(x), reverse_bits_naive128(x));
        CU_ASSERT_EQUAL(reverse_bits128(y), reverse_bits_naive128(y));
        CU_ASSERT_EQUAL(reverse_bits128(z), reverse_bits_naive128(z));
        CU_ASSERT_EQUAL(reverse_bits128(w), reverse_bits_naive128(w));
    }
}

void test_parity128()
{
    for (size_t i = 0; i < sizeof(test_values) / sizeof(test_values[0]); i++)
    {
        CU_ASSERT_EQUAL(parity128(test_values[i]), parity_naive128(test_values[i]));
    }
    for (uint64_t x = 0; x < 1000; x++)
    {
        uint128_t y = UINT128_C2(1234567, x);
        uint128_t z = UINT128_C2(x, 5678901);
        uint128_t w = UINT128_C2(rand(), x);
        CU_ASSERT_EQUAL(parity128(x), parity_naive128(x));
        CU_ASSERT_EQUAL(parity128(y), parity_naive128(y));
        CU_ASSERT_EQUAL(parity128(z), parity_naive128(z));
        CU_ASSERT_EQUAL(parity128(w), parity_naive128(w));
    }
}

void test_is_power_of_2_128()
{
    CU_ASSERT_EQUAL(is_power_of_2_128(0), 0);
    CU_ASSERT_EQUAL(is_power_of_2_128(UINT128_C(1)), 1);
    CU_ASSERT_EQUAL(is_power_of_2_128(UINT128_C(2)), 1);
    CU_ASSERT_EQUAL(is_power_of_2_128(UINT128_C(3)), 0);
    CU_ASSERT_EQUAL(is_power_of_2_128(UINT128_C2(0x8000000000000000, 0)), 1);
    for (uint64_t x = 0; x < 1000; x++)
    {
        uint128_t y = UINT128_C2(1234567, x);
        uint128_t z = UINT128_C2(x, 5678901);
        uint128_t w = UINT128_C2(rand(), x);
        CU_ASSERT_EQUAL(is_power_of_2_128(x), is_power_of_2_naive128(x));
        CU_ASSERT_EQUAL(is_power_of_2_128(y), is_power_of_2_naive128(y));
        CU_ASSERT_EQUAL(is_power_of_2_128(z), is_power_of_2_naive128(z));
        CU_ASSERT_EQUAL(is_power_of_2_128(w), is_power_of_2_naive128(w));
    }
}

void test_rightmost_set_bit128()
{
    for (size_t i = 0; i < sizeof(test_values) / sizeof(test_values[0]); i++)
    {
        CU_ASSERT_EQUAL(rightmost_set_bit128(test_values[i]), rightmost_set_bit_naive128(test_values[i]));
    }
    for (uint64_t x = 1; x < 1000; x++)
    {
        uint128_t y = UINT128_C2(1234567, x);
        uint128_t z = UINT128_C2(x, 5678901);
        uint128_t w = UINT128_C2(rand(), x);
        CU_ASSERT_EQUAL(rightmost_set_bit128(x), rightmost_set_bit_naive128(x));
        CU_ASSERT_EQUAL(rightmost_set_bit128(y), rightmost_set_bit_naive128(y));
        CU_ASSERT_EQUAL(rightmost_set_bit128(z), rightmost_set_bit_naive128(z));
        CU_ASSERT_EQUAL(rightmost_set_bit128(w), rightmost_set_bit_naive128(w));
    }
}

void test_leftmost_set_bit128()
{
    for (size_t i = 0; i < sizeof(test_values) / sizeof(test_values[0]); i++)
    {
        CU_ASSERT_EQUAL(leftmost_set_bit128(test_values[i]), leftmost_set_bit_naive128(test_values[i]));
    }
    for (uint64_t x = 1; x < 1000; x++)
    {
        uint128_t y = UINT128_C2(1234567, x);
        uint128_t z = UINT128_C2(x, 5678901);
        uint128_t w = UINT128_C2(rand(), x);
        CU_ASSERT_EQUAL(leftmost_set_bit128(x), leftmost_set_bit_naive128(x));
        CU_ASSERT_EQUAL(leftmost_set_bit128(y), leftmost_set_bit_naive128(y));
        CU_ASSERT_EQUAL(leftmost_set_bit128(z), leftmost_set_bit_naive128(z));
        CU_ASSERT_EQUAL(leftmost_set_bit128(w), leftmost_set_bit_naive128(w));
    }
}

void test_gray_code128()
{
    CU_ASSERT_EQUAL(gray_code128(0), 0);
    CU_ASSERT_EQUAL(gray_code128(UINT128_C(1)), UINT128_C(1));
    CU_ASSERT_EQUAL(gray_code128(UINT128_C(2)), UINT128_C(3));
    for (uint64_t x = 0; x < 1000; x++)
    {
        uint128_t y = UINT128_C2(1234567, rand());
        uint128_t z = UINT128_C2(rand(), 5678901);
        uint128_t w = UINT128_C2(rand(), 0);
        CU_ASSERT_EQUAL(pop_count128(gray_code128(x) ^ gray_code128(x+1)), 1);
        CU_ASSERT_EQUAL(pop_count128(gray_code128(y) ^ gray_code128(y+1)), 1);
        CU_ASSERT_EQUAL(pop_count128(gray_code128(z) ^ gray_code128(z+1)), 1);
        CU_ASSERT_EQUAL(pop_count128(gray_code128(w) ^ gray_code128(w+1)), 1);
    }
}

void test_inverse_gray_code128()
{
    for (size_t i = 0; i < sizeof(test_values) / sizeof(test_values[0]); i++)
    {
        CU_ASSERT_EQUAL(inverse_gray_code128(gray_code128(test_values[i])), test_values[i]);
    }
    for (uint64_t x = 0; x < 1000; x++)
    {
        uint128_t y = UINT128_C2(1234567, x);
        uint128_t z = UINT128_C2(x, 5678901);
        uint128_t w = UINT128_C2(rand(), x);
        CU_ASSERT_EQUAL(inverse_gray_code128(gray_code128(x)), x);
        CU_ASSERT_EQUAL(inverse_gray_code128(gray_code128(y)), y);
        CU_ASSERT_EQUAL(inverse_gray_code128(gray_code128(z)), z);
        CU_ASSERT_EQUAL(inverse_gray_code128(gray_code128(w)), w);
    }
}

void test_rotate_left128()
{
    CU_ASSERT_EQUAL(rotate_left128(UINT128_C(1), 1), UINT128_C(2));
    CU_ASSERT_EQUAL(rotate_left128(UINT128_C(1), 127), UINT128_C2(0x8000000000000000, 0));
}

void test_rotate_right128()
{
    CU_ASSERT_EQUAL(rotate_right128(UINT128_C(2), 1), UINT128_C(1));
    CU_ASSERT_EQUAL(rotate_right128(UINT128_C2(0x8000000000000000, 0), 127), UINT128_C(1));
}

void test_swap_bits128()
{
    CU_ASSERT_EQUAL(swap_bits128(UINT128_C(5), 0, 2), UINT128_C(5));
    CU_ASSERT_EQUAL(swap_bits128(UINT128_C(5), 0, 1), UINT128_C(6));
    for (uint64_t x = 0; x < 1000; x++)
    {
        uint64_t lo = x;
        uint64_t hi = 1234567 + x + rand();
        uint128_t y = UINT128_C2(hi, lo);
        // test low bits
        CU_ASSERT_EQUAL(swap_bits128(y, 0, 1), UINT128_C2(hi, swap_bits(lo, 0, 1)));
        CU_ASSERT_EQUAL(swap_bits128(y, 1, 2), UINT128_C2(hi, swap_bits(lo, 1, 2)));
        CU_ASSERT_EQUAL(swap_bits128(y, 2, 3), UINT128_C2(hi, swap_bits(lo, 2, 3)));
        CU_ASSERT_EQUAL(swap_bits128(y, 3, 4), UINT128_C2(hi, swap_bits(lo, 3, 4)));

        // test high bits
        CU_ASSERT_EQUAL(swap_bits128(y, 64 + 0, 64 + 1), UINT128_C2(swap_bits(hi, 0, 1), lo));
        CU_ASSERT_EQUAL(swap_bits128(y, 64 + 1, 64 + 2), UINT128_C2(swap_bits(hi, 1, 2), lo));
        CU_ASSERT_EQUAL(swap_bits128(y, 64 + 2, 64 + 3), UINT128_C2(swap_bits(hi, 2, 3), lo));
        CU_ASSERT_EQUAL(swap_bits128(y, 64 + 3, 64 + 4), UINT128_C2(swap_bits(hi, 3, 4), lo));

        // test mixed bits
        CU_ASSERT_EQUAL(swap_bits128(swap_bits128(y, 0, 64 + 1), 0, 64 + 1), y);
        CU_ASSERT_EQUAL(swap_bits128(swap_bits128(y, 1, 64 + 2), 1, 64 + 2), y);
        CU_ASSERT_EQUAL(swap_bits128(swap_bits128(y, 2, 64 + 3), 2, 64 + 3), y);
        CU_ASSERT_EQUAL(swap_bits128(swap_bits128(y, 3, 64 + 4), 3, 64 + 4), y);
        CU_ASSERT_EQUAL(swap_bits128(y, 3, 64 + 7), swap_bits128_naive(y, 3, 64 + 7));
    }
}

int main()
{
    CU_initialize_registry();

    CU_pSuite suite = CU_add_suite("bit_operations128_suite", 0, 0);

    CU_add_test(suite, "test_set_bit128", test_set_bit128);
    CU_add_test(suite, "test_clear_bit128", test_clear_bit128);
    CU_add_test(suite, "test_toggle_bit128", test_toggle_bit128);
    CU_add_test(suite, "test_is_bit_set128", test_is_bit_set128);
    CU_add_test(suite, "test_is_bit_clear128", test_is_bit_clear128);
    CU_add_test(suite, "test_pop_count128", test_pop_count128);
    CU_add_test(suite, "test_bit_block_count128", test_bit_block_count128);
    CU_add_test(suite, "test_reverse_bits128", test_reverse_bits128);
    CU_add_test(suite, "test_parity128", test_parity128);
    CU_add_test(suite, "test_is_power_of_2_128", test_is_power_of_2_128);
    CU_add_test(suite, "test_rightmost_set_bit128", test_rightmost_set_bit128);
    CU_add_test(suite, "test_leftmost_set_bit128", test_leftmost_set_bit128);
    CU_add_test(suite, "test_gray_code128", test_gray_code128);
    CU_add_test(suite, "test_inverse_gray_code128", test_inverse_gray_code128);
    CU_add_test(suite, "test_rotate_left128", test_rotate_left128);
    CU_add_test(suite, "test_rotate_right128", test_rotate_right128);
    CU_add_test(suite, "test_swap_bits128", test_swap_bits128);

    CU_basic_set_mode(CU_BRM_VERBOSE);
    CU_basic_run_tests();
    CU_cleanup_registry();

    return 0;
}
