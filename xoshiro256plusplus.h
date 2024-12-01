
#ifndef XOSHIRO256PLUSPLUS_H
#define XOSHIRO256PLUSPLUS_H

#include <stdint.h>

typedef struct
{
    uint64_t s[4];
} xoshiro256plusplus_t;

void xoshiro256plusplus_seed(xoshiro256plusplus_t *state, uint64_t seed);
uint64_t xoshiro256plusplus_next(xoshiro256plusplus_t *state);
double xoshiro256plusplus_next_double(xoshiro256plusplus_t *state);
void xoshiro256plusplus_jump(xoshiro256plusplus_t *state);
void xoshiro256plusplus_long_jump(xoshiro256plusplus_t *state);

#endif // XOSHIRO256PLUSPLUS_H
