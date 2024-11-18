/**
 * @file hamiltonian.h
 * @brief Header file for hamiltonian.c
 */
#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H

#include <stddef.h>   // size_t
#include <stdint.h>   // uint64_t
#include <petscmat.h> // Mat, PetscErrorCode
#include "bits.h"
#include "bits128.h"
#include "combination.h"

#define MAX_SITE 64 // maximum number of sites

#if MAX_SITE == 64
typedef uint64_t State;
#define print_state(v, n) print_bits(v, n)
#define index2state(n, k, index) index2permutation(n, k, index)
#define state2index(k, state) permutation2index(k, state)
#define next_state(state) next_bit_permutation(state)
#define swap_sites(state, i, j) swap_bits(state, i, j)
#elif MAX_SITE == 128
typedef __uint128_t State;
#define print_state(state, n) print_bits128(state, n)
#define index2state(n, k, index) index2permutation128(n, k, index)
#define state2index(k, state) permutation2index128(k, state)
#define next_state(state) next_bit_permutation128(state)
#define swap_sites(state, i, j) swap_bits128(state, i, j)
#else
#error "MAX_SITE must be 64 or 128"
#endif

// Pair of integers, used to represent a bond
typedef struct Pair
{
    int x;
    int y;
} Pair;

// Context for the simulation
typedef struct Simulation_context
{
    // Configuration, initialized from JSON
    int cnt_site;              // number of sites
    int cnt_bond;              // number of bonds
    int cnt_excitation;        // number of excitations (or number of hardcore bosons)
    Pair *bonds;               // bonds
    double *coupling_strength; // coupling strength
    double total_time;         // total time for the evolution
    int time_steps;            // time steps for the evolution
    State initial_state;       // initial state for the evolution
    State target_state;        // target state for the evolution

    // MPI context
    int n_partition;  // number of partitions
    int partition_id; // partition id of current process

    // Hamiltonian
    size_t h_dimension;           // dimension of the Hilbert space
    Mat hamiltonian;              // Hamiltonian matrix, in sparse form
    size_t local_partition_begin; // the begin index of current partition
    size_t local_partition_size;  // size of the current partition

    // AD context
    Mat *single_bond_hams; // array of single bond Hamiltonians
    Vec init_vec;          // initial state vector
    Vec target_vec;        // target state vector
    Vec *forward_path;     // forward propagation path
    Vec *backward_path;    // backward propagation path

    // output file
    // the format of output is JSON lines (also called newline-delimited JSON), see: https://jsonlines.org/
    FILE *output_file;
} Simulation_context;

// Core initialization and cleanup
PetscErrorCode init_simulation_context(Simulation_context *context, const char *file_name);
PetscErrorCode set_coupling_strength(Simulation_context *context, double *delta);
PetscErrorCode update_coupling_strength(Simulation_context *context, double *coupling_strength);
PetscErrorCode free_simulation_context(Simulation_context *context);

// Matrix building functions
void build_hamiltonian_dense(const Simulation_context *context, double *hamiltonian);
PetscErrorCode build_hamiltonian_sparse(Simulation_context *context, int create_new);
PetscErrorCode build_single_bond_ham_sparse(Simulation_context *context);
PetscErrorCode generate_fock_state(Vec state, State binary_repredentation, const Simulation_context *context);
PetscErrorCode check_Hermitian(Mat A, PetscBool *is_hermitian);

// Utility functions for partitioning
size_t local_partition_size(size_t total_size, size_t n_partition, size_t partition_id);
size_t local_partition_begin(size_t total_size, size_t n_partition, size_t partition_id);

#endif // HAMILTONIAN_H
