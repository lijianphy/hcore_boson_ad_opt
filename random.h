#ifndef RANDOM_H_
#define RANDOM_H_

#include <stdint.h>
#include "splitmix64.h"
#include "xoshiro256plusplus.h"
#include "xoshiro256starstar.h"
#include "rng_config.h"

static inline uint64_t rand_int64(rng_t *rng)
{
    return next(rng);
}

static inline double randu(rng_t *rng)
{
    return next_double(rng);
}

static inline double randu2(rng_t *rng, double a, double b)
{
    return a + (b - a) * randu(rng);
}

double randn(rng_t *rng);
double randn2(rng_t *rng, double mu, double sigma);


#if RNG_HAVE_JUMP
/**
 * Split the RNG into two independent RNGs. Used for parallel computations.
 */
static inline void split_rng(const rng_t *rng, rng_t *new_rng)
{
    *new_rng = *rng;
    jump_forward(new_rng);
}
#endif

#endif // RANDOM_H_
