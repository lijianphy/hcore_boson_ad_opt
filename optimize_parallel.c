#include <stdlib.h> // exit
#include <stdio.h>  // printf
#include <mpi.h>    // MPI_Comm_rank, MPI_COMM_WORLD
#include <petsc.h>  // PetscErrorCode, PetscCall
#include <slepc.h>  // SlepcInitialize
#include <math.h>   // cos, sin
#include <unistd.h> // sleep
#include "log.h"    // print_error_msg_mpi, printf_master
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
    PetscCheckAbort(argc == 2, PETSC_COMM_WORLD, PETSC_ERR_FILE_OPEN, "Usage: %s <config_file>", argv[0]);

    Simulation_context *context = (Simulation_context *)malloc(sizeof(Simulation_context));
    PetscCall(init_simulation_context(context, argv[1]));
    printf_master("Start optimizing the coupling strength in parallel\n");

    PetscCall(optimize_coupling_strength_adam_parallel(context, 0.01, 0.9, 0.999));

    // Clean up
    printf_master("\nCleaning up\n");
    PetscCall(free_simulation_context(context));
    free(context);
    PetscCall(SlepcFinalize());
    return 0;
}
