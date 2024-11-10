/**
 * @file test_vec_math.c
 * @brief Unit tests for vectorized math functions
 */

#include <CUnit/CUnit.h>
#include <CUnit/Basic.h>
#include "vec_math.h"
#include <math.h>
#include "math_constant.h"

void test_vec_sin() {
    double x[] = {0.0, M_PI_2, M_PI};
    double r[3];
    dvec_sin(x, r, 3);
    CU_ASSERT_DOUBLE_EQUAL(r[0], sin(0.0), 0.0001);
    CU_ASSERT_DOUBLE_EQUAL(r[1], sin(M_PI_2), 0.0001);
    CU_ASSERT_DOUBLE_EQUAL(r[2], sin(M_PI), 0.0001);
}

void test_vec_cos() {
    double x[] = {0.0, M_PI_2, M_PI};
    double r[3];
    dvec_cos(x, r, 3);
    CU_ASSERT_DOUBLE_EQUAL(r[0], cos(0.0), 0.0001);
    CU_ASSERT_DOUBLE_EQUAL(r[1], cos(M_PI_2), 0.0001);
    CU_ASSERT_DOUBLE_EQUAL(r[2], cos(M_PI), 0.0001);
}

void test_vec_tan() {
    double x[] = {0.0, M_PI_4, M_PI_2};
    double r[3];
    dvec_tan(x, r, 3);
    CU_ASSERT_DOUBLE_EQUAL(r[0], tan(0.0), 0.0001);
    CU_ASSERT_DOUBLE_EQUAL(r[1], tan(M_PI_4), 0.0001);
    CU_ASSERT_DOUBLE_EQUAL(r[2], tan(M_PI_2), 0.0001);
}


void test_vec_exp() {
    double x[] = {0.0, 1.0, 2.0};
    double r[3];
    dvec_exp(x, r, 3);
    CU_ASSERT_DOUBLE_EQUAL(r[0], exp(0.0), 0.0001);
    CU_ASSERT_DOUBLE_EQUAL(r[1], exp(1.0), 0.0001);
    CU_ASSERT_DOUBLE_EQUAL(r[2], exp(2.0), 0.0001);
}

void test_vec_log() {
    double x[] = {1.0, M_E, 10.0};
    double r[3];
    dvec_log(x, r, 3);
    CU_ASSERT_DOUBLE_EQUAL(r[0], log(1.0), 0.0001);
    CU_ASSERT_DOUBLE_EQUAL(r[1], log(M_E), 0.0001);
    CU_ASSERT_DOUBLE_EQUAL(r[2], log(10.0), 0.0001);
}

int main() {
    CU_initialize_registry();

    CU_pSuite suite = CU_add_suite("vec_math_suite", 0, 0);

    CU_add_test(suite, "test_vec_sin", test_vec_sin);
    CU_add_test(suite, "test_vec_cos", test_vec_cos);
    CU_add_test(suite, "test_vec_tan", test_vec_tan);
    CU_add_test(suite, "test_vec_exp", test_vec_exp);
    CU_add_test(suite, "test_vec_log", test_vec_log);

    CU_basic_set_mode(CU_BRM_VERBOSE);
    CU_basic_run_tests();
    CU_cleanup_registry();

    return 0;
}
