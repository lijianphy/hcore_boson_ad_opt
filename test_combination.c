/**
 * @file test_combination.c
 * @brief Unit tests for combination functions
 */

#include <CUnit/CUnit.h>
#include <CUnit/Basic.h>
#include "combination.h"

static uint64_t factorial_naive(int n) {
    uint64_t ret = 1;
    for (int i = 2; i <= n; i++) {
        ret *= i;
    }
    return ret;
}

void test_gcd() {
    CU_ASSERT_EQUAL(gcd(48, 18), 6);
    CU_ASSERT_EQUAL(gcd(101, 103), 1);
    CU_ASSERT_EQUAL(gcd(56, 98), 14);
}

void test_factorial() {
    CU_ASSERT_EQUAL(factorial(0), 1);
    CU_ASSERT_EQUAL(factorial(1), 1);
    CU_ASSERT_EQUAL(factorial(5), 120);
    CU_ASSERT_EQUAL(factorial(10), 3628800);
    for(int i = 0; i <= 20; i++) {
        CU_ASSERT_EQUAL(factorial(i), factorial_naive(i));
    }
}

void test_binomial() {
    CU_ASSERT_EQUAL(binomial(5, 2), 10);
    CU_ASSERT_EQUAL(binomial(6, 3), 20);
    CU_ASSERT_EQUAL(binomial(10, 5), 252);
    CU_ASSERT_EQUAL(binomial(10, 0), 1);
    CU_ASSERT_EQUAL(binomial(10, 10), 1);
    CU_ASSERT_EQUAL(binomial(30, 10), 30045015);
    CU_ASSERT_EQUAL(binomial(30, 15), 155117520);
}

void test_bit_permutation() {
    int n = 23;
    int k = 5;
    uint64_t index = 0;
    uint64_t data = (UINT64_C(1) << k) - 1;

    uint64_t max_index = binomial(n, k);
    while (index < max_index) {
        CU_ASSERT_EQUAL(index2permutation(n, k, index), data);
        CU_ASSERT_EQUAL(permutation2index(k, data), index);
        data = next_bit_permutation(data);
        index++;
    }
}


int main() {
    CU_initialize_registry();

    CU_pSuite suite = CU_add_suite("combination_suite", 0, 0);

    CU_add_test(suite, "test_gcd", test_gcd);
    CU_add_test(suite, "test_factorial", test_factorial);
    CU_add_test(suite, "test_binomial", test_binomial);
    CU_add_test(suite, "test_bit_permutation", test_bit_permutation);

    CU_basic_set_mode(CU_BRM_VERBOSE);
    CU_basic_run_tests();
    CU_cleanup_registry();

    return 0;
}
