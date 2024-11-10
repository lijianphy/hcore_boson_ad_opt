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

    Simulation_context context = {0}; // Initialize all fields to 0
    PetscCall(init_simulation_context(&context, argv[1]));

    printf_master("Random initialize coupling strength\n");
    double norm2_grad;
    double threshold = 1e-3;
    PetscCall(random_initialize_coupling_strength(&context, 100, threshold, &norm2_grad));
    if (norm2_grad < threshold)
    {
        printf_master("WARNING: Gradient is small after random initialization\n");
    }


    printf_master("Start optimizing the coupling strength\n");

    // run the optimization using gradient descent
    // PetscCall(optimize_coupling_strength_gd(&context, 1000, 0.01));

    // run the optimization using Adam optimizer
    // Generally, Adam optimizer converges faster than gradient descent
    PetscCall(optimize_coupling_strength_adam(&context, 1000, 0.01, 0.9, 0.999));

    // print the optimized coupling strength
    printf_master("Optimized coupling strength:\n");
    for (int i = 0; i < context.cnt_bond; i++)
    {
        printf_master("%lf", context.coupling_strength[i]);
        if (i < context.cnt_bond - 1)
        {
            printf_master(", ");
        }
    }
    printf_master("\n");

    // Clean up
    printf_master("\nCleaning up\n");
    PetscCall(free_simulation_context(&context));
    PetscCall(SlepcFinalize());
    return 0;
}
