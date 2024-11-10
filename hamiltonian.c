/**
 * @file hamiltonian.c
 * @brief Implementation of Hamiltonian matrix operations for hardcore boson simulations
 *
 * This file contains functions for handling both sparse and dense representations
 * of Hamiltonian matrices used in hardcore boson simulations. It includes functionality
 * for reading configuration from a JSON file, validating the configuration, 
 * initializing the simulation context, building Hamiltonian matrices, and managing
 * the simulation state.
 *
 * Key functionalities:
 * - Reading and validating JSON configuration files
 * - Initializing simulation context and Hamiltonian matrices
 * - Building sparse and dense Hamiltonian matrices
 * - Managing simulation states and paths
 * - Utility functions for partitioning and matrix operations
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <cJSON.h>
#include <petscmat.h>

#include "bits.h"        // next_bit_permutation, swap_bits
#include "combination.h" // binomial, index2permutation, permutation2index, next_bit_permutation
#include "log.h"         // print_error_msg, print_error_msg_mpi, printf_master, printf_mpi

#include "hamiltonian.h"

/**
 * @brief Validates the JSON schema for configuration
 * @param json The JSON object to validate
 * @return 1 if the JSON is valid, 0 otherwise
 */
static int validate_json_schema(const cJSON *json)
{
    // Check required fields exist
    const char *required_fields[] = {"cnt_site", "cnt_bond", "cnt_excitation", "bonds", "coupling_strength",
                                     "total_time", "time_steps", "initial_state", "target_state"};
    size_t num_required_fields = sizeof(required_fields) / sizeof(required_fields[0]);
    for (size_t i = 0; i < num_required_fields; i++)
    {
        cJSON *item = cJSON_GetObjectItem(json, required_fields[i]);
        if (item == NULL)
        {
            print_error_msg_mpi("Missing required field: %s", required_fields[i]);
            return 0;
        }
    }

    /// Check types and values
    // Check numbers
    const char *number_fields[] = {"cnt_site", "cnt_bond", "cnt_excitation", "total_time", "time_steps"};
    size_t num_number_fields = sizeof(number_fields) / sizeof(number_fields[0]);
    for (size_t i = 0; i < num_number_fields; i++)
    {
        if (!cJSON_IsNumber(cJSON_GetObjectItem(json, number_fields[i])))
        {
            print_error_msg_mpi("%s must be a number", number_fields[i]);
            return 0;
        }
    }

    // Check number of sites and excitations
    int cnt_site = cJSON_GetObjectItem(json, "cnt_site")->valueint;
    int cnt_bond = cJSON_GetObjectItem(json, "cnt_bond")->valueint;
    int cnt_excitation = cJSON_GetObjectItem(json, "cnt_excitation")->valueint;
    // cnt_site should be less than 64
    if (cnt_site >= 64)
    {
        print_error_msg_mpi("cnt_site must be less than 64, got %d", cnt_site);
        return 0;
    }
    // cnt_excitation should be > 0 and < cnt_site
    if (cnt_excitation <= 0 || cnt_excitation >= cnt_site)
    {
        print_error_msg_mpi("cnt_excitation must be > 0 and < cnt_site, got %d", cnt_excitation);
        return 0;
    }

    // Check arrays
    const char *array_fields[] = {"bonds", "coupling_strength", "initial_state", "target_state"};
    const int array_sizes[] = {cnt_bond, cnt_bond, cnt_excitation, cnt_excitation};
    size_t num_array_fields = sizeof(array_fields) / sizeof(array_fields[0]);
    for (size_t i = 0; i < num_array_fields; i++)
    {
        cJSON *array = cJSON_GetObjectItem(json, array_fields[i]);
        if (!cJSON_IsArray(array))
        {
            print_error_msg_mpi("%s must be an array", array_fields[i]);
            return 0;
        }
        if (cJSON_GetArraySize(array) != array_sizes[i])
        {
            print_error_msg_mpi("%s array size must be %d", array_fields[i], array_sizes[i]);
            return 0;
        }
    }

    // Validate each bond
    cJSON *bonds = cJSON_GetObjectItem(json, "bonds");
    cJSON *strengths = cJSON_GetObjectItem(json, "coupling_strength");
    for (int i = 0; i < cnt_bond; i++)
    {
        cJSON *bond = cJSON_GetArrayItem(bonds, i);
        if (!cJSON_IsArray(bond) || cJSON_GetArraySize(bond) != 2 ||
            !cJSON_IsNumber(cJSON_GetArrayItem(bond, 0)) ||
            !cJSON_IsNumber(cJSON_GetArrayItem(bond, 1)))
        {
            print_error_msg_mpi("Each bond must be an array of 2 integers");
            return 0;
        }
        // first and second elements of bond should be between 0 and cnt_site - 1
        int site1 = cJSON_GetArrayItem(bond, 0)->valueint;
        int site2 = cJSON_GetArrayItem(bond, 1)->valueint;
        if (site1 < 0 || site1 >= cnt_site || site2 < 0 || site2 >= cnt_site)
        {
            print_error_msg_mpi("Elements of bonds must be between 0 and cnt_site - 1");
            return 0;
        }
        if (!cJSON_IsNumber(cJSON_GetArrayItem(strengths, i)))
        {
            print_error_msg_mpi("Each coupling strength must be a number");
            return 0;
        }
    }

    // Elements of bonds should be unique
    for (int i = 0; i < cnt_bond; i++)
    {
        for (int j = i + 1; j < cnt_bond; j++)
        {
            cJSON *bond1 = cJSON_GetArrayItem(bonds, i);
            cJSON *bond2 = cJSON_GetArrayItem(bonds, j);

            int x1 = cJSON_GetArrayItem(bond1, 0)->valueint;
            int y1 = cJSON_GetArrayItem(bond1, 1)->valueint;
            int x2 = cJSON_GetArrayItem(bond2, 0)->valueint;
            int y2 = cJSON_GetArrayItem(bond2, 1)->valueint;

            if ((x1 == x2 && y1 == y2) || (x1 == y2 && y1 == x2))
            {
                print_error_msg_mpi("Elements of bonds must be unique");
                return 0;
            }
        }
    }

    // Validate initial and target states
    cJSON *initial_state = cJSON_GetObjectItem(json, "initial_state");
    cJSON *target_state = cJSON_GetObjectItem(json, "target_state");
    // array elements should be integers and between 0 and cnt_site - 1
    for (int i = 0; i < cnt_excitation; i++)
    {
        if (!cJSON_IsNumber(cJSON_GetArrayItem(initial_state, i)) ||
            !cJSON_IsNumber(cJSON_GetArrayItem(target_state, i)))
        {
            print_error_msg_mpi("Initial and target states must be arrays of integers");
            return 0;
        }
        int initial_site = cJSON_GetArrayItem(initial_state, i)->valueint;
        int target_site = cJSON_GetArrayItem(target_state, i)->valueint;
        if (initial_site < 0 || initial_site >= cnt_site || target_site < 0 || target_site >= cnt_site)
        {
            print_error_msg_mpi("Initial and target states must be between 0 and cnt_site - 1");
            return 0;
        }
    }

    // elements of initial state and target state should be unique
    for (int i = 0; i < cnt_excitation; i++)
    {
        for (int j = i + 1; j < cnt_excitation; j++)
        {
            if (cJSON_GetArrayItem(initial_state, i)->valueint == cJSON_GetArrayItem(initial_state, j)->valueint ||
                cJSON_GetArrayItem(target_state, i)->valueint == cJSON_GetArrayItem(target_state, j)->valueint)
            {
                print_error_msg_mpi("Elements of initial and target states must be unique");
                return 0;
            }
        }
    }

    return 1;
}

/**
 * @brief Reads configuration from a JSON file and sets the configuration in the context
 * @param file_name Path to the configuration JSON file
 * @param context Pointer to the simulation context to be populated
 * @throw Exits with error code 1 if file operations or JSON parsing fails
 */
static void read_config(const char *file_name, Simulation_context *context)
{
    // Read the entire file into a string
    FILE *file = fopen(file_name, "r");
    if (file == NULL)
    {
        print_error_msg_mpi("Unable to open file %s", file_name);
        exit(1);
    }

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    rewind(file);

    char *json_str = (char *)malloc(file_size + 1);
    if (json_str == NULL)
    {
        print_error_msg_mpi("Unable to allocate memory");
        exit(1);
    }
    long s = fread(json_str, 1, file_size, file);
    if (s != file_size)
    {
        print_error_msg_mpi("Error reading file %s", file_name);
        free(json_str);
        exit(1);
    }
    json_str[file_size] = '\0';
    fclose(file);

    // Parse JSON
    cJSON *json = cJSON_Parse(json_str);
    if (json == NULL)
    {
        const char *error_ptr = cJSON_GetErrorPtr();
        if (error_ptr != NULL)
        {
            print_error_msg_mpi("Error parsing JSON: %s", error_ptr);
        }
        free(json_str);
        exit(1);
    }

    // Validate JSON schema
    if (!validate_json_schema(json))
    {
        cJSON_Delete(json);
        free(json_str);
        exit(1);
    }

    // Extract configuration
    context->cnt_site = cJSON_GetObjectItem(json, "cnt_site")->valueint;
    context->cnt_bond = cJSON_GetObjectItem(json, "cnt_bond")->valueint;
    context->cnt_excitation = cJSON_GetObjectItem(json, "cnt_excitation")->valueint;

    // Read bonds and coupling_strength
    context->bonds = (Pair *)malloc(context->cnt_bond * sizeof(Pair));
    context->coupling_strength = (double *)malloc(context->cnt_bond * sizeof(double));

    cJSON *bonds = cJSON_GetObjectItem(json, "bonds");
    cJSON *strengths = cJSON_GetObjectItem(json, "coupling_strength");
    for (int i = 0; i < context->cnt_bond; i++)
    {
        cJSON *bond = cJSON_GetArrayItem(bonds, i);
        context->bonds[i].x = cJSON_GetArrayItem(bond, 0)->valueint;
        context->bonds[i].y = cJSON_GetArrayItem(bond, 1)->valueint;
        context->coupling_strength[i] = cJSON_GetArrayItem(strengths, i)->valuedouble;
    }

    // Read total_time, time_steps, initial_state, target_state
    context->total_time = cJSON_GetObjectItem(json, "total_time")->valuedouble;
    context->time_steps = cJSON_GetObjectItem(json, "time_steps")->valueint;
    context->initial_state = 0;
    context->target_state = 0;
    cJSON *initial_state = cJSON_GetObjectItem(json, "initial_state");
    cJSON *target_state = cJSON_GetObjectItem(json, "target_state");
    for (int i = 0; i < context->cnt_excitation; i++)
    {
        context->initial_state |= 1ULL << cJSON_GetArrayItem(initial_state, i)->valueint;
        context->target_state |= 1ULL << cJSON_GetArrayItem(target_state, i)->valueint;
    }

    // Cleanup
    cJSON_Delete(json);
    free(json_str);
}

/**
 * @brief Prints the current configuration settings
 * @param context Pointer to the simulation context containing the configuration
 * @return Always returns 0
 */
static int print_config(const Simulation_context *context)
{
    printf("cnt_site: %d\n", context->cnt_site);
    printf("cnt_bond: %d\n", context->cnt_bond);
    printf("cnt_excitation: %d\n", context->cnt_excitation);

    for (int i = 0; i < context->cnt_bond; i++)
    {
        printf("bond[%d]: %d %d %lf\n", i, context->bonds[i].x, context->bonds[i].y, context->coupling_strength[i]);
    }

    printf("total_time: %lf\n", context->total_time);
    printf("time_steps: %d\n", context->time_steps);
    printf("initial_state: ");
    print_bits(context->initial_state, context->cnt_site);
    printf("target_state:  ");
    print_bits(context->target_state, context->cnt_site);

    return 0;
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
    // get n_partition and partition_id
    int n_partition = 0;
    int partition_id = 0;
    MPI_Comm_size(PETSC_COMM_WORLD, &n_partition);
    MPI_Comm_rank(PETSC_COMM_WORLD, &partition_id);
    context->n_partition = n_partition;
    context->partition_id = partition_id;
    printf_master("Number of partitions: %d\n", n_partition);

    printf_master("Reading configuration from %s\n", file_name);
    read_config(file_name, context);

    printf_master("Got configuration:\n");
    if (partition_id == 0)
    {
        print_config(context);
    }

    // calculate the dimension of the Hilbert space
    size_t h_dimension = binomial(context->cnt_site, context->cnt_excitation);
    context->h_dimension = h_dimension;
    printf_master("Hilbert space dimension: %zu\n", h_dimension);

    // calculate the local size and begin index
    context->local_partition_size = local_partition_size(h_dimension, n_partition, partition_id);
    context->local_partition_begin = local_partition_begin(h_dimension, n_partition, partition_id);

    printf_master("Building sparse Hamiltonian matrix\n");
    PetscCall(build_hamiltonian_sparse(context, 1));

    printf_master("Build single bond Hamiltonians\n");
    PetscCall(build_single_bond_ham_sparse(context));

    printf_master("Build initial and target vectors\n");
    PetscCall(VecCreateMPI(PETSC_COMM_WORLD, context->local_partition_size, h_dimension, &context->init_vec));
    PetscCall(VecDuplicate(context->init_vec, &context->target_vec));
    PetscCall(generate_fock_state(context->init_vec, context->initial_state, context));
    PetscCall(generate_fock_state(context->target_vec, context->target_state, context));

    // allocate forward and backward paths
    printf_master("Allocate and init forward and backward paths\n");
    PetscCall(VecDuplicateVecs(context->init_vec, context->time_steps + 1, &context->forward_path));
    PetscCall(VecDuplicateVecs(context->init_vec, context->time_steps + 1, &context->backward_path));
    PetscCall(VecCopy(context->init_vec, context->forward_path[0]));

    return PETSC_SUCCESS;
}

/**
 * @brief Set the coupling strength in the simulation context and rebuilds the Hamiltonian matrix
 * @param context Pointer to the simulation context
 * @param coupling_strength Array of new coupling strengths
 * @return PetscErrorCode
 */
PetscErrorCode set_coupling_strength(Simulation_context *context, double *coupling_strength)
{
    for (int i = 0; i < context->cnt_bond; i++)
    {
        context->coupling_strength[i] = coupling_strength[i];
    }
    PetscCall(build_hamiltonian_sparse(context, 0));
    return PETSC_SUCCESS;
}

/**
 * @brief Updates the coupling strength in the simulation context and rebuilds the Hamiltonian matrix
 * @param context Pointer to the simulation context
 * @param delta Array of delta coupling strengths
 * @return PetscErrorCode
 */
PetscErrorCode update_coupling_strength(Simulation_context *context, double *delta)
{
    for (int i = 0; i < context->cnt_bond; i++)
    {
        context->coupling_strength[i] += delta[i];
    }
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
    // free hamiltonian
    PetscCall(MatDestroy(&context->hamiltonian));

    // free single bond Hamiltonians
    for (int i = 0; i < context->cnt_bond; i++)
    {
        PetscCall(MatDestroy(&context->single_bond_hams[i]));
    }
    free(context->single_bond_hams);

    // free initial and target vectors
    PetscCall(VecDestroy(&context->init_vec));
    PetscCall(VecDestroy(&context->target_vec));

    // free forward and backward paths
    PetscCall(VecDestroyVecs(context->time_steps + 1, &context->forward_path));
    PetscCall(VecDestroyVecs(context->time_steps + 1, &context->backward_path));

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

    uint64_t current_state = index2permutation(cnt_site, cnt_excitation, 0);
    for (size_t i = 0; i < h_dimension; i++, current_state = next_bit_permutation(current_state))
    {
        double *p = hamiltonian + i * h_dimension;
        for (int j = 0; j < cnt_bond; j++)
        {
            Pair bond = bonds[j];
            uint64_t new_state = swap_bits(current_state, bond.x, bond.y);
            if (new_state != current_state)
            {
                size_t new_index = permutation2index(cnt_excitation, new_state);
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

    uint64_t current_state = index2permutation(cnt_site, cnt_excitation, partition_begin);
    for (size_t i = 0; i < partition_size; i++, current_state = next_bit_permutation(current_state))
    {
        for (int j = 0; j < cnt_bond; j++)
        {
            Pair bond = bonds[j];
            uint64_t new_state = swap_bits(current_state, bond.x, bond.y);
            if (new_state != current_state)
            {
                size_t new_index = permutation2index(cnt_excitation, new_state);
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
        PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, mpi_local_size, mpi_local_size, h_dimension,
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
    uint64_t current_state = index2permutation(cnt_site, cnt_excitation, mpi_local_begin);
    for (size_t i = mpi_local_begin; i < mpi_local_end; i++, current_state = next_bit_permutation(current_state))
    {
        PetscInt current_row = i;
        PetscInt cnt_values = 0;
        for (int j = 0; j < cnt_bond; j++)
        {
            Pair bond = bonds[j];
            uint64_t new_state = swap_bits(current_state, bond.x, bond.y);
            if (new_state != current_state)
            {
                size_t new_index = permutation2index(cnt_excitation, new_state);
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
    PetscCheckAbort(is_hermitian == PETSC_TRUE, PETSC_COMM_WORLD, 1, "Hamiltonian matrix is not Hermitian");
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
        Pair bond = bonds[j];

        // create the matrix
        PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, mpi_local_size, mpi_local_size, h_dimension,
                               h_dimension, 1, NULL, 1, NULL, &single_bond_hams[j]));

        PetscCall(MatSetUp(single_bond_hams[j]));

        // fill the matrix
        uint64_t current_state = index2permutation(cnt_site, cnt_excitation, mpi_local_begin);
        for (size_t i = mpi_local_begin; i < mpi_local_end; i++, current_state = next_bit_permutation(current_state))
        {
            uint64_t new_state = swap_bits(current_state, bond.x, bond.y);
            if (new_state != current_state)
            {
                size_t new_index = permutation2index(cnt_excitation, new_state);
                PetscCall(MatSetValue(single_bond_hams[j], i, new_index, -1.0 * I, INSERT_VALUES));
            }
        }
        PetscCall(MatAssemblyBegin(single_bond_hams[j], MAT_FINAL_ASSEMBLY));
    }

    for (int j = 0; j < cnt_bond; j++)
    {
        PetscCall(MatAssemblyEnd(single_bond_hams[j], MAT_FINAL_ASSEMBLY));
    }
    return PETSC_SUCCESS;
}

/**
 * @brief Generates a Fock state vector from a binary representation
 * @param state The vector to store the Fock state
 * @param binary_repredentation The binary representation of the Fock state
 * @param context Pointer to the simulation context
 */
PetscErrorCode generate_fock_state(Vec state, uint64_t binary_repredentation, const Simulation_context *context)
{
    size_t mpi_local_begin = context->local_partition_begin;
    size_t mpi_local_end = context->local_partition_begin + context->local_partition_size;

    PetscCall(VecSet(state, 0.0));

    size_t index = permutation2index(context->cnt_excitation, binary_repredentation);
    if ((index >= mpi_local_begin) && (index < mpi_local_end))
    {
        PetscCall(VecSetValue(state, index, 1.0, INSERT_VALUES));
    }
    PetscCall(VecAssemblyBegin(state));
    PetscCall(VecAssemblyEnd(state));
    return PETSC_SUCCESS;
}
