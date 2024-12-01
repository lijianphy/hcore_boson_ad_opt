/**
 * @file test_optimize.c
 * @brief Test the optimization of the coupling strength using gradient descent and Adam optimizer
 */

#include <stdlib.h>   // exit
#include <stdio.h>    // printf
#include <mpi.h>      // MPI_Comm_rank, MPI_COMM_WORLD
#include <petsc.h>    // PetscErrorCode, PetscCall
#include <slepc.h>    // SlepcInitialize
#include "log.h"      // print_error_msg_mpi, printf_master
#include "hamiltonian.h"
#include "evolution_ad.h"

// optimize the coupling strength using gradient descent
int main(int argc, char *argv[])
{
    const char *help = "Random sampling the coupling strength \n";
    // Initialize PETSc and SLEPc
    PetscCall(SlepcInitialize(&argc, &argv, NULL, help));

    if (argc != 2)
    {
        print_error_msg_mpi("Usage: %s <config_file>", argv[0]);
        exit(1);
    }

    Simulation_context* context = (Simulation_context*)malloc(sizeof(Simulation_context));
    PetscCall(init_simulation_context(context, argv[1]));
    printf_master("Start sampling the coupling strength\n");
    PetscCall(random_sampling_coupling_strength(context, 10000, 0.05, 5.0));

    // Clean up
    printf_master("\nCleaning up\n");
    PetscCall(free_simulation_context(context));
    free(context);
    PetscCall(SlepcFinalize());
    return 0;
}
