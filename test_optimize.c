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
    const char *help = "Optimize the coupling strength using gradient descent\n";
    // Initialize PETSc and SLEPc
    PetscCall(SlepcInitialize(&argc, &argv, NULL, help));
    if (argc != 2)
    {
        print_error_msg_mpi("Usage: %s <config_file>", argv[0]);
        exit(1);
    }

    Simulation_context *context = (Simulation_context *)malloc(sizeof(Simulation_context));
    PetscCall(init_simulation_context(context, argv[1]));

    printf_master("Start optimizing the coupling strength\n");

    // run the optimization using Adam optimizer
    // Generally, Adam optimizer converges faster than gradient descent
    PetscCall(optimize_coupling_strength_adam(context, context->total_iteration, 0.01, 0.9, 0.999, AD_V2));

    // print the optimized coupling strength
    if (context->partition_id == 0)
    {
        printf("Optimized coupling strength of stream %d:\n", context->stream_id);
        for (int i = 0; i < context->cnt_bond; i++)
        {
            printf("%lf", context->coupling_strength[i]);
            if (i < context->cnt_bond - 1)
            {
                printf(", ");
            }
        }
        printf("\n");
    }

    // Clean up
    printf_master("\nCleaning up\n");
    PetscCall(free_simulation_context(context));
    free(context);
    PetscCall(SlepcFinalize());
    return 0;
}
