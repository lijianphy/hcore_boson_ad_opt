/**
 * @file test_cblas.c
 * @brief Unit tests for cblas functions
 */

#include <CUnit/CUnit.h>
#include <CUnit/Basic.h>
#include <cblas.h>

void test_cblas_sasum() {
    float x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    CU_ASSERT_DOUBLE_EQUAL(cblas_sasum(5, x, 1), 15.0, 1e-10);
}

void test_cblas_dasum() {
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    CU_ASSERT_DOUBLE_EQUAL(cblas_dasum(5, x, 1), 15.0, 1e-10);
}


int main(void) {
    CU_initialize_registry();

    CU_pSuite suite = CU_add_suite("cblas_suite", 0, 0);

    CU_add_test(suite, "test_cblas_sasum", test_cblas_sasum);
    CU_add_test(suite, "test_cblas_dasum", test_cblas_dasum);

    CU_basic_set_mode(CU_BRM_VERBOSE);
    CU_basic_run_tests();
    CU_cleanup_registry();

    return 0;
}
