/*  Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide.

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR
IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE. */

#include <stdint.h>
#include "splitmix64.h"
#include "xoshiro256starstar.h"

static inline uint64_t rotl(const uint64_t x, int k)
{
    return (x << k) | (x >> (64 - k));
}

void xoshiro256starstar_seed(xoshiro256starstar_t *state, uint64_t seed)
{
    splitmix64_t sm64;
    splitmix64_seed(&sm64, seed);

    state->s[0] = splitmix64_next(&sm64);
    state->s[1] = splitmix64_next(&sm64);
    state->s[2] = splitmix64_next(&sm64);
    state->s[3] = splitmix64_next(&sm64);
}

uint64_t xoshiro256starstar_next(xoshiro256starstar_t *state)
{
    uint64_t *s = state->s;
    const uint64_t result = rotl(s[1] * 5, 7) * 9;

    const uint64_t t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;

    s[3] = rotl(s[3], 45);

    return result;
}

void xoshiro256starstar_jump(xoshiro256starstar_t *state)
{
    static const uint64_t JUMP[] = { 0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c };

    uint64_t *s = state->s;
    uint64_t s0 = 0;
    uint64_t s1 = 0;
    uint64_t s2 = 0;
    uint64_t s3 = 0;
    for(int i = 0; i < 4; i++)
        for(int b = 0; b < 64; b++) {
            if (JUMP[i] & UINT64_C(1) << b) {
                s0 ^= s[0];
                s1 ^= s[1];
                s2 ^= s[2];
                s3 ^= s[3];
            }
            xoshiro256starstar_next(state);
        }
        
    s[0] = s0;
    s[1] = s1;
    s[2] = s2;
    s[3] = s3;
}

void xoshiro256starstar_long_jump(xoshiro256starstar_t *state)
{
    static const uint64_t LONG_JUMP[] = { 0x76e15d3efefdcbbf, 0xc5004e441c522fb3, 0x77710069854ee241, 0x39109bb02acbe635 };

    uint64_t *s = state->s;
    uint64_t s0 = 0;
    uint64_t s1 = 0;
    uint64_t s2 = 0;
    uint64_t s3 = 0;
    for(int i = 0; i < 4; i++)
        for(int b = 0; b < 64; b++) {
            if (LONG_JUMP[i] & UINT64_C(1) << b) {
                s0 ^= s[0];
                s1 ^= s[1];
                s2 ^= s[2];
                s3 ^= s[3];
            }
            xoshiro256starstar_next(state);
        }
        
    s[0] = s0;
    s[1] = s1;
    s[2] = s2;
    s[3] = s3;
}

double xoshiro256starstar_next_double(xoshiro256starstar_t *state)
{
    return (xoshiro256starstar_next(state) >> 11) * 0x1.0p-53;
}
