/**
 * @file log.h
 * @brief This file contains macros for logging messages. This is a header only library.
 */

#ifndef LOG_H_
#define LOG_H_

#include <stdio.h> // fprintf, printf
#include <mpi.h>   // MPI_Comm_rank, MPI_COMM_WORLD

// Macro that prints error messages with file name,
// line number and function name information
#define print_error_msg(...)                                                    \
    do                                                                          \
    {                                                                           \
        fprintf(stderr, "[%s:%d in %s] Error: ", __FILE__, __LINE__, __func__); \
        fprintf(stderr, __VA_ARGS__);                                           \
        fprintf(stderr, "\n");                                                  \
    } while (0)

// Macro that prints error messages for each rank in MPI, with rank information
#define print_error_msg_mpi(...)                                                                       \
    do                                                                                                 \
    {                                                                                                  \
        int __mpi_rank;                                                                                \
        MPI_Comm_rank(MPI_COMM_WORLD, &__mpi_rank);                                                    \
        fprintf(stderr, "[%s:%d in %s] Error in rank %d: ", __FILE__, __LINE__, __func__, __mpi_rank); \
        fprintf(stderr, __VA_ARGS__);                                                                  \
        fprintf(stderr, "\n");                                                                         \
    } while (0)

// Macro that prints only for rank 0 (master) in MPI
#define printf_master(...)                          \
    do                                              \
    {                                               \
        int __mpi_rank;                             \
        MPI_Comm_rank(MPI_COMM_WORLD, &__mpi_rank); \
        if (__mpi_rank == 0)                        \
        {                                           \
            printf(__VA_ARGS__);                    \
        }                                           \
    } while (0)

// Macro that prints for each rank in MPI
#define printf_mpi(...)                             \
    do                                              \
    {                                               \
        int __mpi_rank;                             \
        MPI_Comm_rank(MPI_COMM_WORLD, &__mpi_rank); \
        printf("[rank %d] ", __mpi_rank);           \
        printf(__VA_ARGS__);                        \
    } while (0)

#endif // LOG_H_
