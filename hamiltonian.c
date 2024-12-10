/**
 * @file hamiltonian.c
 * @brief Implementation of Hamiltonian matrix operations for hardcore boson simulations
 *
 * This file contains functions for handling both sparse and dense representations
 * of Hamiltonian matrices used in hardcore boson simulations. It includes functionality
 * for initializing the simulation context, building Hamiltonian matrices, and managing
 * the simulation state.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <stdarg.h>
#include <cJSON.h>
#include <petscmat.h>
#include "log.h"
#include "hamiltonian.h"
#include "simu_config.h"

/**
 * @brief Generate output file name from input file name and current time
 * @param file_name Input file name
 * @return Output file name
 */
static char *generate_output_file_name(const char *file_name)
{
    // Extract base name from path
    const char *base_name = strrchr(file_name, '/');
    base_name = base_name ? base_name + 1 : file_name;

    // Find last dot position and calculate base length
    const char *last_dot = strrchr(base_name, '.');
    size_t base_len = last_dot ? (size_t)(last_dot - base_name) : strlen(base_name);

    // Get current time
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);

    // Generate random hex string
    char random_str[9]; // 8 chars + null terminator
    unsigned int random_val;
    FILE *f = fopen("/dev/urandom", "rb");
    if (f)
    {
        size_t cnt_read = fread(&random_val, sizeof(random_val), 1, f);
        fclose(f);
        if (cnt_read != 1)
        {
            random_val = (unsigned int)time(NULL) ^ (unsigned int)clock();
        }
    }
    else
    {
        random_val = (unsigned int)time(NULL) ^ (unsigned int)clock();
    }
    snprintf(random_str, sizeof(random_str), "%08x", random_val);

    // Generate output filename with timestamp and random string
    char *output_file = (char *)malloc(base_len + 48); // enough space for name + random + timestamp + .jsonl

    // Copy base name (without extension) and add random string and timestamp
    memcpy(output_file, base_name, base_len);
    output_file[base_len] = '\0';

    // output file name format: base_name_RANDOM_YYYYMMDD-HHMMSS.jsonl
    sprintf(output_file + base_len, "_%04d%02d%02d-%02d%02d%02d_%s.jsonl",
            tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
            tm.tm_hour, tm.tm_min, tm.tm_sec, random_str);

    return output_file;
}

/**
 * @brief Initializes the simulation context by setting up configuration and Hamiltonian matrix
 * @param context Pointer to the simulation context to be initialized
 * @param file_name Path to the configuration JSON file
 * @return PetscErrorCode
 *
 * This function:
 * 1. Reads configuration from a JSON file
 * 2. Calculates the dimension of the Hilbert space
 * 3. Calculates the local size and begin index of the partition
 * 4. Builds the sparse Hamiltonian matrix
 */
PetscErrorCode init_simulation_context(Simulation_context *context, const char *file_name)
{
    // clear the context
    memset(context, 0, sizeof(Simulation_context));

    // get total_rank and rank_id
    int total_rank = 0;
    int rank_id = 0;
    MPI_Comm_size(PETSC_COMM_WORLD, &total_rank);
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank_id);
    printf_master("Number of process: %d\n", total_rank);

    printf_master("Reading configuration from %s\n", file_name);
    PetscCall(read_config(file_name, context));

    printf_master("Got configuration:\n");
    if (rank_id == 0)
    {
        print_config(context);
    }

    // check if the number of processes is equal to n_streams * n_partition
    if (total_rank != (context->n_streams * context->n_partition))
    {
        print_error_msg_mpi("Number of processes must be equal to n_streams * n_partition");
        return PETSC_ERR_ARG_WRONG;
    }

    // split the communicator into streams
    int color = rank_id / context->n_partition;
    PetscCall(MPI_Comm_split(PETSC_COMM_WORLD, color, rank_id, &context->comm));
    context->stream_id = color;
    PetscCall(MPI_Comm_rank(context->comm, &context->partition_id));

    // differenciate the master process from the others
    color = context->partition_id == 0 ? 0 : 1;
    PetscCall(MPI_Comm_split(PETSC_COMM_WORLD, color, rank_id, &context->master_comm));
    PetscCall(MPI_Comm_rank(context->master_comm, &context->master_rank));
    context->is_master = context->partition_id == 0 ? 1 : 0;
    printf("Rank %d: stream_id %d, partition_id %d, master_rank %d, is_master %d\n",
           rank_id, context->stream_id, context->partition_id, context->master_rank, context->is_master);
    
    // broadcast root process id to all processes
    int root_id = context->is_master && (context->master_rank == 0) ? rank_id : -1;
    PetscCall(MPI_Allreduce(&root_id, &context->root_id, 1, MPI_INT, MPI_MAX, PETSC_COMM_WORLD));

    // open output file
    // the format of output is JSON lines (also called newline-delimited JSON), see: https://jsonlines.org/
    context->output_file = NULL;
    if (context->partition_id == 0)
    {
        char *output_file = generate_output_file_name(file_name);
        context->output_file = fopen(output_file, "w");
        if (context->output_file == NULL)
        {
            print_error_msg_mpi("Unable to open file %s for writing", output_file);
            free(output_file);
            return PETSC_ERR_FILE_OPEN;
        }
        printf_master("Output file: %s\n", output_file);
        free(output_file);
    }

    // calculate the dimension of the Hilbert space
    size_t h_dimension = binomial(context->cnt_site, context->cnt_excitation);
    context->h_dimension = h_dimension;
    printf_master("Hilbert space dimension: %zu\n", h_dimension);

    // calculate the local size and begin index
    context->local_partition_size = local_partition_size(h_dimension, context->n_partition, context->partition_id);
    context->local_partition_begin = local_partition_begin(h_dimension, context->n_partition, context->partition_id);

    printf_master("Building sparse Hamiltonian matrix\n");
    PetscCall(build_hamiltonian_sparse(context, 1));

    printf_master("Build single bond Hamiltonians\n");
    PetscCall(build_single_bond_ham_sparse(context));

    printf_master("Build initial and target vectors\n");
    PetscCall(VecCreateMPI(context->comm, context->local_partition_size, h_dimension, &context->init_vec));
    PetscCall(VecDuplicate(context->init_vec, &context->target_vec));
    PetscCall(generate_fock_state(context->init_vec, context->initial_state, context));
    PetscCall(generate_fock_state(context->target_vec, context->target_state, context));

    // allocate forward and backward paths
    printf_master("Allocate and init forward and backward paths\n");
    PetscCall(VecDuplicateVecs(context->init_vec, context->time_steps + 1, &context->forward_path));
    PetscCall(VecDuplicateVecs(context->init_vec, context->time_steps + 1, &context->backward_path));
    PetscCall(VecCopy(context->init_vec, context->forward_path[0]));

    // initalize random number generator
    context->rng = (rng_t *)malloc(sizeof(rng_t));
    set_seed(context->rng, context->rng_seed);
    for (int i = 0; i < context->stream_id; i++) // set different random stream for each stream
    {
        jump_forward(context->rng);
    }

    return PETSC_SUCCESS;
}

/**
 * @brief Set the coupling strength in the simulation context and rebuilds the Hamiltonian matrix
 * @param context Pointer to the simulation context
 * @param coupling_strength Array of new coupling strengths
 * @return PetscErrorCode
 *
 * For consistency, the coupling_strength array should be the same across all partitions.
 * Thus only the master process sets the coupling strength, and then broadcasts it to all processes.
 */
PetscErrorCode set_coupling_strength(Simulation_context *context, double *coupling_strength)
{
    // only the master process sets the coupling strength
    if (context->partition_id == 0)
    {
        for (int i = 0; i < context->cnt_bond; i++)
        {
            context->coupling_strength[i] = coupling_strength[i];
        }
    }
    // broadcast the coupling strength to all processes
    PetscCall(MPI_Bcast(context->coupling_strength, context->cnt_bond, MPI_DOUBLE, 0, context->comm));
    PetscCall(build_hamiltonian_sparse(context, 0));
    return PETSC_SUCCESS;
}

/**
 * @brief Updates the coupling strength in the simulation context and rebuilds the Hamiltonian matrix
 * @param context Pointer to the simulation context
 * @param delta Array of delta coupling strengths
 * @return PetscErrorCode
 *
 */
PetscErrorCode update_coupling_strength(Simulation_context *context, double *delta)
{
    // only the master process updates the coupling strength
    if (context->partition_id == 0)
    {
        for (int i = 0; i < context->cnt_bond; i++)
        {
            context->coupling_strength[i] += delta[i];
        }
    }
    // broadcast the coupling strength to all processes
    PetscCall(MPI_Bcast(context->coupling_strength, context->cnt_bond, MPI_DOUBLE, 0, context->comm));
    PetscCall(build_hamiltonian_sparse(context, 0));
    return PETSC_SUCCESS;
}

/**
 * @brief Frees all allocated memory in the simulation context
 * @param context Pointer to the simulation context to be freed
 * @return PetscErrorCode
 */
PetscErrorCode free_simulation_context(Simulation_context *context)
{
    // free bonds and coupling_strength
    free(context->bonds);
    free(context->coupling_strength);
    free(context->isfixed);
    // free hamiltonian
    PetscCall(MatDestroy(&context->hamiltonian));

    // free single bond Hamiltonians
    for (int i = 0; i < context->cnt_bond; i++)
    {
        if (!context->isfixed[i])
        {
            PetscCall(MatDestroy(&context->single_bond_hams[i]));
        }
    }
    free(context->single_bond_hams);

    // free initial and target vectors
    PetscCall(VecDestroy(&context->init_vec));
    PetscCall(VecDestroy(&context->target_vec));

    // free forward and backward paths
    PetscCall(VecDestroyVecs(context->time_steps + 1, &context->forward_path));
    PetscCall(VecDestroyVecs(context->time_steps + 1, &context->backward_path));

    // close output file
    if (context->output_file != NULL)
    {
        fclose(context->output_file);
    }

    // free random number generator
    free(context->rng);

    // free MPI communicator
    PetscCall(MPI_Comm_free(&context->comm));

    return PETSC_SUCCESS;
}

/**
 * @brief Builds the Hamiltonian matrix in dense form
 * @param context Pointer to the simulation context
 * @param hamiltonian Pointer to pre-allocated memory for the dense matrix
 *
 * The matrix is stored in row-major order.
 */
void build_hamiltonian_dense(const Simulation_context *context, double *hamiltonian)
{
    // extract configuration
    int cnt_site = context->cnt_site;
    int cnt_excitation = context->cnt_excitation;
    int cnt_bond = context->cnt_bond;
    size_t h_dimension = context->h_dimension;
    Pair *bonds = context->bonds;
    double *coupling_strength = context->coupling_strength;

    // clear the hamiltonian
    memset(hamiltonian, 0, h_dimension * h_dimension * sizeof(double));

    State current_state = index2state(cnt_site, cnt_excitation, 0);
    for (size_t i = 0; i < h_dimension; i++, current_state = next_state(current_state))
    {
        double *p = hamiltonian + i * h_dimension;
        for (int j = 0; j < cnt_bond; j++)
        {
            Pair bond = bonds[j];
            State new_state = swap_sites(current_state, bond.x, bond.y);
            if (new_state != current_state)
            {
                size_t new_index = state2index(cnt_excitation, new_state);
                p[new_index] += coupling_strength[j];
            }
        }
    }
}

/**
 * @brief Calculates the local size of a partition
 * @param total_size Total size to be partitioned
 * @param n_partition Number of partitions
 * @param partition_id Partition ID (0-based)
 * @return Size of the specified partition
 */
size_t local_partition_size(size_t total_size, size_t n_partition, size_t partition_id)
{
    size_t local_size = total_size / n_partition;
    size_t rem = total_size % n_partition;
    if (partition_id < rem)
    {
        return local_size + 1;
    }
    else
    {
        return local_size;
    }
}

/**
 * @brief Calculates the local begin index of a partition
 * @param total_size Total size to be partitioned
 * @param n_partition Number of partitions
 * @param partition_id Partition ID (0-based)
 * @return Begin index of the specified partition
 */
size_t local_partition_begin(size_t total_size, size_t n_partition, size_t partition_id)
{
    size_t local_size = total_size / n_partition;
    size_t rem = total_size % n_partition;
    if (partition_id < rem)
    {
        return partition_id * (local_size + 1);
    }
    else
    {
        return total_size - local_size * (n_partition - partition_id);
    }
}

/**
 * @brief Initializes the d_nnz and o_nnz arrays for creating the sparse matrix in PETSc
 * @param context Pointer to the simulation context
 * @param dnnz Array to store the number of non-zeros in the diagonal rows
 * @param onnz Array to store the number of non-zeros in off the diagonal rows
 *
 * These arrays are used to create the sparse matrix in PETSc.
 * See: https://petsc.org/release/manualpages/Mat/MatMPIAIJSetPreallocation/
 */
static void init_dnnz_onnz(Simulation_context *context, PetscInt *dnnz, PetscInt *onnz)
{
    // extract configuration
    int cnt_site = context->cnt_site;
    int cnt_excitation = context->cnt_excitation;
    int cnt_bond = context->cnt_bond;
    Pair *bonds = context->bonds;
    size_t partition_begin = context->local_partition_begin;
    size_t partition_size = context->local_partition_size;
    size_t partition_end = partition_begin + partition_size;

    // clear the arrays
    memset(dnnz, 0, partition_size * sizeof(PetscInt));
    memset(onnz, 0, partition_size * sizeof(PetscInt));

    State current_state = index2state(cnt_site, cnt_excitation, partition_begin);
    for (size_t i = 0; i < partition_size; i++, current_state = next_state(current_state))
    {
        for (int j = 0; j < cnt_bond; j++)
        {
            Pair bond = bonds[j];
            State new_state = swap_sites(current_state, bond.x, bond.y);
            if (new_state != current_state)
            {
                size_t new_index = state2index(cnt_excitation, new_state);
                if (new_index >= partition_begin && new_index < partition_end)
                {
                    dnnz[i]++;
                }
                else
                {
                    onnz[i]++;
                }
            }
        }
    }
}

/**
 * @brief Checks if a matrix is Hermitian
 * @param A Matrix to check
 * @param is_hermitian Pointer to store the result
 * @return PetscErrorCode
 */
PetscErrorCode check_Hermitian(Mat A, PetscBool *is_hermitian)
{
    Mat AT; // Transpose matrix

#if defined(PETSC_USE_COMPLEX)
    PetscCall(MatCreateHermitianTranspose(A, &AT));
#else
    PetscCall(MatCreateTranspose(A, &AT));
#endif
    // Check if A equals its transpose
    PetscCall(MatEqual(A, AT, is_hermitian));
    // Cleanup
    PetscCall(MatDestroy(&AT));

    return PETSC_SUCCESS;
}

/**
 * @brief Builds the Hamiltonian matrix in sparse form
 * @param context Pointer to the simulation context
 * @param create_new Whether to create a new matrix or not, if not, the matrix is zeroed
 * @return PetscErrorCode
 */
PetscErrorCode build_hamiltonian_sparse(Simulation_context *context, int create_new)
{
    // extract configuration
    int cnt_site = context->cnt_site;
    int cnt_excitation = context->cnt_excitation;
    int cnt_bond = context->cnt_bond;
    size_t h_dimension = context->h_dimension;
    Pair *bonds = context->bonds;
    double *coupling_strength = context->coupling_strength;

    // get the local size and begin index
    size_t mpi_local_size = context->local_partition_size;
    size_t mpi_local_begin = context->local_partition_begin;
    size_t mpi_local_end = mpi_local_begin + mpi_local_size;

    if (create_new)
    {
        // initialize d_nnz and o_nnz
        PetscInt *d_nnz = (PetscInt *)malloc(mpi_local_size * sizeof(PetscInt));
        PetscInt *o_nnz = (PetscInt *)malloc(mpi_local_size * sizeof(PetscInt));
        init_dnnz_onnz(context, d_nnz, o_nnz);
        PetscCall(MatCreateAIJ(context->comm, mpi_local_size, mpi_local_size, h_dimension,
                               h_dimension, 0, d_nnz, 0, o_nnz, &context->hamiltonian));

        PetscCall(MatSetUp(context->hamiltonian));
        free(d_nnz);
        free(o_nnz);
    }
    else
    {
        PetscCall(MatZeroEntries(context->hamiltonian));
    }

    // fill the matrix
    PetscInt *index_buffer = (PetscInt *)malloc(cnt_bond * sizeof(PetscInt));
    PetscScalar *value_buffer = (PetscScalar *)malloc(cnt_bond * sizeof(PetscScalar));
    State current_state = index2state(cnt_site, cnt_excitation, mpi_local_begin);
    for (size_t i = mpi_local_begin; i < mpi_local_end; i++, current_state = next_state(current_state))
    {
        PetscInt current_row = i;
        PetscInt cnt_values = 0;
        for (int j = 0; j < cnt_bond; j++)
        {
            Pair bond = bonds[j];
            State new_state = swap_sites(current_state, bond.x, bond.y);
            if (new_state != current_state)
            {
                size_t new_index = state2index(cnt_excitation, new_state);
                index_buffer[cnt_values] = new_index;
                value_buffer[cnt_values] = coupling_strength[j];
                cnt_values++;
            }
        }
        PetscCall(MatSetValues(context->hamiltonian, 1, &current_row, cnt_values, index_buffer, value_buffer, INSERT_VALUES));
    }
    free(index_buffer);
    free(value_buffer);

    PetscCall(MatAssemblyBegin(context->hamiltonian, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(context->hamiltonian, MAT_FINAL_ASSEMBLY));

#ifdef DEBUG
    // check if the matrix is Hermitian
    PetscBool is_hermitian;
    PetscCall(check_Hermitian(context->hamiltonian, &is_hermitian));
    PetscCheckAbort(is_hermitian == PETSC_TRUE, context->comm, 1, "Hamiltonian matrix is not Hermitian");
#endif

    return PETSC_SUCCESS;
}

/**
 * @brief Builds the single bond Hamiltonian for each bond in sparse form
 * @param context Pointer to the simulation context
 * @return PetscErrorCode
 */
PetscErrorCode build_single_bond_ham_sparse(Simulation_context *context)
{
    // extract configuration
    int cnt_site = context->cnt_site;
    int cnt_excitation = context->cnt_excitation;
    int cnt_bond = context->cnt_bond;
    size_t h_dimension = context->h_dimension;
    Pair *bonds = context->bonds;

    // allocate memory for single bond Hamiltonians
    context->single_bond_hams = (Mat *)malloc(cnt_bond * sizeof(Mat));
    Mat *single_bond_hams = context->single_bond_hams;

    // get the local size and begin index
    size_t mpi_local_size = context->local_partition_size;
    size_t mpi_local_begin = context->local_partition_begin;
    size_t mpi_local_end = mpi_local_begin + mpi_local_size;

    for (int j = 0; j < cnt_bond; j++)
    {
        if (context->isfixed[j]) // only build single bond Hamiltonian for non-fixed bonds
        {
            context->single_bond_hams[j] = NULL;
            continue;
        }

        Pair bond = bonds[j];

        // create the matrix
        PetscCall(MatCreateAIJ(context->comm, mpi_local_size, mpi_local_size, h_dimension,
                               h_dimension, 1, NULL, 1, NULL, &single_bond_hams[j]));

        PetscCall(MatSetUp(single_bond_hams[j]));

        // fill the matrix
        State current_state = index2state(cnt_site, cnt_excitation, mpi_local_begin);
        for (size_t i = mpi_local_begin; i < mpi_local_end; i++, current_state = next_state(current_state))
        {
            State new_state = swap_sites(current_state, bond.x, bond.y);
            if (new_state != current_state)
            {
                size_t new_index = state2index(cnt_excitation, new_state);
                PetscCall(MatSetValue(single_bond_hams[j], i, new_index, -1.0 * I, INSERT_VALUES));
            }
        }
        PetscCall(MatAssemblyBegin(single_bond_hams[j], MAT_FINAL_ASSEMBLY));
    }

    for (int j = 0; j < cnt_bond; j++)
    {
        if (!context->isfixed[j]) // only build single bond Hamiltonian for non-fixed bonds
        {
            PetscCall(MatAssemblyEnd(single_bond_hams[j], MAT_FINAL_ASSEMBLY));
        }
    }
    return PETSC_SUCCESS;
}

/**
 * @brief Generates a Fock state vector from a binary representation
 * @param state The vector to store the Fock state
 * @param binary_repredentation The binary representation of the Fock state
 * @param context Pointer to the simulation context
 */
PetscErrorCode generate_fock_state(Vec state, State binary_repredentation, const Simulation_context *context)
{
    size_t mpi_local_begin = context->local_partition_begin;
    size_t mpi_local_end = context->local_partition_begin + context->local_partition_size;

    PetscCall(VecSet(state, 0.0));

    size_t index = state2index(context->cnt_excitation, binary_repredentation);
    if ((index >= mpi_local_begin) && (index < mpi_local_end))
    {
        PetscCall(VecSetValue(state, index, 1.0, INSERT_VALUES));
    }
    PetscCall(VecAssemblyBegin(state));
    PetscCall(VecAssemblyEnd(state));
    return PETSC_SUCCESS;
}

/**
 * @brief Calculates the fidelity between two normalized vectors
 * @param vec1 First vector
 * @param vec2 Second vector
 * @param result Pointer to store the fidelity
 * @return PetscErrorCode
 */
PetscErrorCode calc_fidelity(Vec vec1, Vec vec2, double *result)
{
    PetscScalar inner_product;
    PetscCall(VecDot(vec1, vec2, &inner_product));
    double abs_inner_product = PetscAbsScalar(inner_product);
    *result = abs_inner_product * abs_inner_product;
    return PETSC_SUCCESS;
}
