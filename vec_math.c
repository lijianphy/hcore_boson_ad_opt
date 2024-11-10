/**
 * @file vec_math.c
 * @brief Vectorized math functions
 */
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <cblas.h>

/**
 * Set all elements of x to value
 */
void dvec_set(double *x, double value, size_t n)
{
    if (value == 0.0)
    {
        memset(x, 0, n * sizeof(double));
    }
    else
    {
        for (size_t i = 0; i < n; i++)
        {
            x[i] = value;
        }
    }
}

/**
 * Copy the elements of x to r
 * x and r should not overlap
 */
void dvec_copy(const double *restrict x, double *restrict r, size_t n)
{
    memcpy(r, x, n * sizeof(double));
}

/**
 * Pack the elements of x using the indices array and store the result in r
 */
void dvec_pack(const double *restrict x, double *restrict r, const size_t *restrict indices, size_t m)
{
    for (size_t i = 0; i < m; i++)
    {
        r[i] = x[indices[i]];
    }
}

/**
 * Unpack the elements of x using the indices array and store the result in r
 */
void dvec_unpack(const double *restrict x, double *restrict r, const size_t *restrict indices, size_t m)
{
    for (size_t i = 0; i < m; i++)
    {
        r[indices[i]] = x[i];
    }
}

/**
 * Pack the slice of x and store the result in r
 */
void dvec_pack_slice(const double *restrict x, double *restrict r, size_t step, size_t m)
{
    cblas_dcopy(m, x, step, r, 1);
}

/**
 * Unpack the slice of x and store the result in r
 */
void dvec_unpack_slice(const double *restrict x, double *restrict r, size_t step, size_t m)
{
    cblas_dcopy(m, x, 1, r, step);
}

/**
 * Helper function for qsort
 */
static int compare_double(const void *a, const void *b)
{
    double x = *(double *)a;
    double y = *(double *)b;
    if (x < y)
    {
        return -1;
    }
    else if (x > y)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

/**
 * Sort the elements of x in place
 */
void dvec_sort(double *x, size_t n)
{
    qsort(x, n, sizeof(double), compare_double);
}

/**
 * Check if the elements of x are sorted
 */
int dvec_issorted(const double *x, size_t n)
{
    for (size_t i = 1; i < n; i++)
    {
        if (x[i - 1] > x[i])
        {
            return 0;
        }
    }
    return 1;
}

/**
 * Compute the sine of each element in x and store the result in r
 * x and r should not overlap
 */
void dvec_sin(const double *restrict x, double *restrict r, size_t n)
{
#pragma omp simd
    for (size_t i = 0; i < n; i++)
    {
        r[i] = sin(x[i]);
    }
}

/**
 * Compute the sine of each element in x inplace (the result is stored in x)
 */
void dvec_sin_inplace(double *x, size_t n)
{
#pragma omp simd
    for (size_t i = 0; i < n; i++)
    {
        x[i] = sin(x[i]);
    }
}

/**
 * Compute the cosine of each element in x and store the result in r
 */
void dvec_cos(const double *restrict x, double *restrict r, size_t n)
{
#pragma omp simd
    for (size_t i = 0; i < n; i++)
    {
        r[i] = cos(x[i]);
    }
}

/**
 * Compute the cosine of each element in x inplace (the result is stored in x)
 */
void dvec_cos_inplace(double *x, size_t n)
{
#pragma omp simd
    for (size_t i = 0; i < n; i++)
    {
        x[i] = cos(x[i]);
    }
}

/**
 * Compute the tangent of each element in x and store the result in r
 */
void dvec_tan(const double *restrict x, double *restrict r, size_t n)
{
#pragma omp simd
    for (size_t i = 0; i < n; i++)
    {
        r[i] = tan(x[i]);
    }
}

/**
 * Compute the tangent of each element in x inplace (the result is stored in x)
 */
void dvec_tan_inplace(double *x, size_t n)
{
#pragma omp simd
    for (size_t i = 0; i < n; i++)
    {
        x[i] = tan(x[i]);
    }
}

/**
 * Compute the exp of each element in x and store the result in r
 */
void dvec_exp(const double *restrict x, double *restrict r, size_t n)
{
#pragma omp simd
    for (size_t i = 0; i < n; i++)
    {
        r[i] = exp(x[i]);
    }
}

/**
 * Compute the exp of each element in x inplace (the result is stored in x)
 */
void dvec_exp_inplace(double *x, size_t n)
{
#pragma omp simd
    for (size_t i = 0; i < n; i++)
    {
        x[i] = exp(x[i]);
    }
}

/**
 * Compute the log of each element in x and store the result in r
 */
void dvec_log(const double *restrict x, double *restrict r, size_t n)
{
#pragma omp simd
    for (size_t i = 0; i < n; i++)
    {
        r[i] = log(x[i]);
    }
}

/**
 * Compute the log of each element in x inplace (the result is stored in x)
 */
void dvec_log_inplace(double *x, size_t n)
{
#pragma omp simd
    for (size_t i = 0; i < n; i++)
    {
        x[i] = log(x[i]);
    }
}

/**
 * add alpha to each element in x and store the result in r
 */
void dvec_add_scalar(const double *restrict x, double *restrict r, double alpha, size_t n)
{
#pragma omp simd
    for (size_t i = 0; i < n; i++)
    {
        r[i] = x[i] + alpha;
    }
}

/**
 * multiply each element in x by alpha and store the result in r
 */
void dvec_mul_scalar(const double *restrict x, double *restrict r, double alpha, size_t n)
{
#pragma omp simd
    for (size_t i = 0; i < n; i++)
    {
        r[i] = x[i] * alpha;
    }
}

/**
 * add alpha to each element in x inplace (the result is stored in x)
 */
void dvec_add_scalar_inplace(double *x, double alpha, size_t n)
{
#pragma omp simd
    for (size_t i = 0; i < n; i++)
    {
        x[i] += alpha;
    }
}

/**
 * multiply each element in x by alpha inplace (the result is stored in x)
 */
void dvec_mul_scalar_inplace(double *x, double alpha, size_t n)
{
    cblas_dscal(n, alpha, x, 1);
}

/**
 * Generate a vector of linearly spaced values
 */
void dvec_linspace(double start, double stop, size_t num, double *result)
{
    if (num == 0)
    {
        return;
    }
    if (num == 1)
    {
        result[0] = start;
        return;
    }

    // num > 1
    double step = (stop - start) / (double)(num - 1);
    for (size_t i = 0; i < (num - 1); i++)
    {
        result[i] = start + i * step;
    }
    result[num - 1] = stop;
    return;
}

/**
 * Generate a vector of random values between 0 and 1
 */
void dvec_rand(double *x, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        x[i] = (double)rand() / (double)RAND_MAX;
    }
}
