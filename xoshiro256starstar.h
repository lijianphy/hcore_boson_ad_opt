#ifndef XOSHIRO256STARSTAR_H
#define XOSHIRO256STARSTAR_H

#include <stdint.h>

typedef struct
{
    uint64_t s[4];
} xoshiro256starstar_t;

void xoshiro256starstar_seed(xoshiro256starstar_t *state, uint64_t seed);
uint64_t xoshiro256starstar_next(xoshiro256starstar_t *state);
double xoshiro256starstar_next_double(xoshiro256starstar_t *state);
void xoshiro256starstar_jump(xoshiro256starstar_t *state);
void xoshiro256starstar_long_jump(xoshiro256starstar_t *state);

#endif // XOSHIRO256STARSTAR_H
