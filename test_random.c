#include <stdio.h>
#include "random.h"

int main(void)
{
    rng_t rng;
    printf("sizeof(rng_t) = %lu\n", sizeof(rng_t));
    set_seed(&rng, 1234);
    for (int i = 0; i < 10; i++)
    {
        printf("%.10f\n", randu(&rng));
    }

// split the RNG into two independent RNGs
#if RNG_HAVE_JUMP
    rng_t rng2;
    split_rng(&rng, &rng2);
    printf("\nNew rng stream:\n");
    for (int i = 0; i < 10; i++)
    {
        printf("%.10f\n", randu(&rng2));
    }
#endif

    return 0;
}
