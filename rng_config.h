#ifndef RNG_CONFIG_H_
#define RNG_CONFIG_H_

// This is the RNG type you want to use
// supported RNGs: xoshiro256starstar, xoshiro256plusplus, splitmix64
// where xoshiro256starstar and xoshiro256plusplus support jump operations
#define RNG_TYPE xoshiro256plusplus
// whether the RNG supports jump operations
#define RNG_HAVE_JUMP 1

// Define the RNG type
#define RNG_T_X(name) name ## _t
#define RNG_T(name) RNG_T_X(name)
#define rng_t RNG_T(RNG_TYPE)

// Generic RNG operation macro
#define RNG_OP_X(T, op, ...) T ## _ ## op(__VA_ARGS__)
#define RNG_OP(T, op, ...) RNG_OP_X(T, op, __VA_ARGS__)

// RNG operations
#define set_seed(rng, s) RNG_OP(RNG_TYPE, seed, rng, s)
#define next(rng) RNG_OP(RNG_TYPE, next, rng)
#define next_double(rng) RNG_OP(RNG_TYPE, next_double, rng)

// RNG jump operations
#if RNG_HAVE_JUMP
#define jump_forward(rng) RNG_OP(RNG_TYPE, jump, rng)
#endif

#endif // RNG_CONFIG_H_
