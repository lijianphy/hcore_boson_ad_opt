#include <math.h>
#include "random.h"

/**
 * Generate a random number from a standard normal distribution.
 * See: Knuth v2, 3rd ed, p122
 */
double randn(rng_t *rng)
{
    double x, y, r2;

    do
    {
        /* choose x,y in uniform square (-1,-1) to (+1,+1) */
        x = -1.0 + 2.0 * next_double(rng);
        y = -1.0 + 2.0 * next_double(rng);

        /* see if it is in the unit circle */
        r2 = x * x + y * y;
    } while (r2 > 1.0 || r2 == 0);

    /* Box-Muller transform */
    return y * sqrt(-2.0 * log(r2) / r2);
}

/**
 * Generate a random number from a normal distribution with mean mu and standard deviation sigma.
 */
double randn2(rng_t *rng, double mu, double sigma)
{
    return mu + sigma * randn(rng);
}


