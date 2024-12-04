#include <stdlib.h>   // exit
#include <stdio.h>    // printf
#include <mpi.h>      // MPI_Comm_rank, MPI_COMM_WORLD
#include <petsc.h>    // PetscErrorCode, PetscCall
#include <slepc.h>    // SlepcInitialize
#include <math.h>     // cos, sin
#include <unistd.h>   // sleep
#include "log.h"      // print_error_msg_mpi, printf_master
#include "hamiltonian.h"
#include "evolution_ad.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// optimize the coupling strength using gradient descent
int main(int argc, char *argv[])
{
    const char *help = "Optimize the coupling strength using parallel Adam\n";
    // Initialize PETSc and SLEPc
    PetscCall(SlepcInitialize(&argc, &argv, NULL, help));
    if (argc != 2)
    {
        print_error_msg_mpi("Usage: %s <config_file>", argv[0]);
        exit(1);
    }

    int cnt_parallel = 5;
    PetscScalar *phi_list = (PetscScalar *)malloc(cnt_parallel * sizeof(PetscScalar));
    for (int i = 0; i < cnt_parallel; i++)
    {
        double phi = 2.0 * M_PI * i / cnt_parallel;
        phi_list[i] = cos(phi) + I * sin(phi);
    }

    Simulation_context *context_list = (Simulation_context *)malloc(cnt_parallel * sizeof(Simulation_context));
    for (int i = 0; i < cnt_parallel; i++)
    {
        PetscCall(init_simulation_context(context_list + i, argv[1]));
    }

    printf_master("Start optimizing the coupling strength in parallel\n");
    PetscCall(optimize_coupling_strength_adam_parallel(context_list, cnt_parallel, phi_list, 10000, 0.01, 0.9, 0.999));

    // Clean up
    printf_master("\nCleaning up\n");
    for (int i = 0; i < cnt_parallel; i++)
    {
        PetscCall(free_simulation_context(context_list + i));
    }
    free(context_list);
    PetscCall(SlepcFinalize());
    return 0;
}
