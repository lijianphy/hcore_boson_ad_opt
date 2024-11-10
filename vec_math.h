#ifndef VEC_MATH_H
#define VEC_MATH_H

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Set all elements of x to value
 */
void dvec_set(double *x, double value, size_t n);

/**
 * Copy the elements of x to r
 * x and r should not overlap
 */
void dvec_copy(const double *restrict x, double *restrict r, size_t n);

/**
 * Pack the elements of x using the indices array and store the result in r
 */
void dvec_pack(const double *restrict x, double *restrict r, const size_t *restrict indices, size_t m);

/**
 * Unpack the elements of x using the indices array and store the result in r
 */
void dvec_unpack(const double *restrict x, double *restrict r, const size_t *restrict indices, size_t m);

/**
 * Pack the slice of x and store the result in r
 */
void dvec_pack_slice(const double *restrict x, double *restrict r, size_t step, size_t m);

/**
 * Unpack the slice of x and store the result in r
 */
void dvec_unpack_slice(const double *restrict x, double *restrict r, size_t step, size_t m);

/**
 * Sort the elements of x in place
 */
void dvec_sort(double *x, size_t n);

/**
 * Check if the elements of x are sorted
 */
int dvec_issorted(const double *x, size_t n);

/**
 * Compute the sine of each element in x and store the result in r
 * x and r should not overlap
 */
void dvec_sin(const double *restrict x, double *restrict r, size_t n);

/**
 * Compute the sine of each element in x inplace (the result is stored in x)
 */
void dvec_sin_inplace(double *x, size_t n);

/**
 * Compute the cosine of each element in x and store the result in r
 */
void dvec_cos(const double *restrict x, double *restrict r, size_t n);

/**
 * Compute the cosine of each element in x inplace (the result is stored in x)
 */
void dvec_cos_inplace(double *x, size_t n);

/**
 * Compute the tangent of each element in x and store the result in r
 */
void dvec_tan(const double *restrict x, double *restrict r, size_t n);

/**
 * Compute the tangent of each element in x inplace (the result is stored in x)
 */
void dvec_tan_inplace(double *x, size_t n);

/**
 * Compute the exp of each element in x and store the result in r
 */
void dvec_exp(const double *restrict x, double *restrict r, size_t n);

/**
 * Compute the exp of each element in x inplace (the result is stored in x)
 */
void dvec_exp_inplace(double *x, size_t n);

/**
 * Compute the log of each element in x and store the result in r
 */
void dvec_log(const double *restrict x, double *restrict r, size_t n);

/**
 * Compute the log of each element in x inplace (the result is stored in x)
 */
void dvec_log_inplace(double *x, size_t n);

/**
 * add alpha to each element in x and store the result in r
 */
void dvec_add_scalar(const double *restrict x, double *restrict r, double alpha, size_t n);

/**
 * multiply each element in x by alpha and store the result in r
 */
void dvec_mul_scalar(const double *restrict x, double *restrict r, double alpha, size_t n);

/**
 * add alpha to each element in x inplace (the result is stored in x)
 */
void dvec_add_scalar_inplace(double *x, double alpha, size_t n);

/**
 * multiply each element in x by alpha inplace (the result is stored in x)
 */
void dvec_mul_scalar_inplace(double *x, double alpha, size_t n);

/**
 * Generate a vector of linearly spaced values
 */
void dvec_linspace(double start, double stop, size_t num, double *result);

/**
 * Generate a vector of random values between 0 and 1
 */
void dvec_rand(double *x, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* VEC_MATH_H */
