/**
 * @file test_ad.c
 * @brief This file contains the main function to calculate the differentiation by Automatic Differentiation (AD).
 *        It initializes the simulation context, runs the evolution, calculates the gradient,
 *        and compares it with finite differences.
 */

#include <stdlib.h>   // malloc, free
#include <lapacke.h>  // LAPACKE_dsyev
#include <petscmat.h> // Mat
#include <slepceps.h> // EPS
#include "log.h"      // print_error_msg_mpi, printf_master
#include "hamiltonian.h"
#include "evolution_ad.h"

static const AD_TYPE ad_type = AD_V3;

static PetscErrorCode vec_diff_norm(Vec vec1, Vec vec2, double *norm)
{
    Vec diff;
    PetscCall(VecDuplicate(vec1, &diff));
    PetscCall(VecWAXPY(diff, -1.0, vec2, vec1));
    PetscCall(VecNorm(diff, NORM_2, norm));
    PetscCall(VecDestroy(&diff));
    return PETSC_SUCCESS;
}

int main(int argc, char *argv[])
{
    const char *help = "Calculate the differentiation by AD\n";
    // Initialize PETSc and SLEPc
    PetscCall(SlepcInitialize(&argc, &argv, NULL, help));

    if (argc != 2)
    {
        print_error_msg_mpi("Usage: %s <config_file>", argv[0]);
        exit(1);
    }

    Simulation_context context = {0}; // Initialize all fields to 0
    PetscCall(init_simulation_context(&context, argv[1]));

    // allocate memory for gradient
    double *grad = (double *)malloc(context.cnt_bond * sizeof(double));
    if (grad == NULL)
    {
        print_error_msg_mpi("Unable to allocate memory for gradient");
        exit(1);
    }
    // run the evolution and calculate the gradient
    printf_master("Running evolution\n");
    PetscErrorCode (*run_evolution)(Simulation_context *) = NULL;
    switch (ad_type)
    {
    case AD_V1:
        run_evolution = run_evolution_v1;
        break;
    case AD_V2:
        run_evolution = run_evolution_v2;
        break;
    case AD_V3:
        run_evolution = run_evolution_v3;
        break;
    default:
        print_error_msg_mpi("Invalid AD type");
        exit(1);
    }

    PetscCall(run_evolution(&context));
    PetscCall(calculate_gradient(&context, grad));

    // print the gradient
    if (context.partition_id == 0)
    {
        printf("Gradient from AD:\n");
        for (int i = 0; i < context.cnt_bond; i++)
        {
            printf("Bond %2d: %+.6lf\n", i, grad[i]);
        }
    }

    double fidelity1, fidelity2;
    double diff_norm1, diff_norm2;

    // Calculate the norm of the difference between the forward path at the last time step and the target vector
    PetscCall(vec_diff_norm(context.forward_path[context.time_steps], context.target_vec, &diff_norm1));
    PetscCall(calc_fidelity(context.forward_path[context.time_steps], context.target_vec, &fidelity1));

    // Update the coupling strength
    double *initial_coupling_strength = (double *)malloc(context.cnt_bond * sizeof(double));
    double *new_coupling_strength = (double *)malloc(context.cnt_bond * sizeof(double));
    if (initial_coupling_strength == NULL || new_coupling_strength == NULL)
    {
        print_error_msg_mpi("Unable to allocate memory for initial or new coupling strength");
        exit(1);
    }

    for (int i = 0; i < context.cnt_bond; i++)
    {
        initial_coupling_strength[i] = context.coupling_strength[i];
    }

    double delta = 1e-6;

    printf_master("\nGradient from finite difference:\n");
    for (int i = 0; i < context.cnt_bond; i++)
    {
        for (int j = 0; j < context.cnt_bond; j++)
        {
            new_coupling_strength[j] = initial_coupling_strength[j];
        }
        if (!context.isfixed[i])
        {
            new_coupling_strength[i] += delta;
        }
        PetscCall(set_coupling_strength(&context, new_coupling_strength));
        PetscCall(forward_evolution(&context));
        PetscCall(vec_diff_norm(context.forward_path[context.time_steps], context.target_vec, &diff_norm2));
        PetscCall(calc_fidelity(context.forward_path[context.time_steps], context.target_vec, &fidelity2));
        double finite_diff_norm2 = (diff_norm2 * diff_norm2 - diff_norm1 * diff_norm1) / delta;
        double finite_diff_fedelity = (fidelity1 - fidelity2) / delta;
        double finite_diff = (ad_type == AD_V1) ? finite_diff_norm2 : finite_diff_fedelity;
        double relative_error = (grad[i] - finite_diff) / finite_diff;
        printf_master("Bond %2d: %+.6lf, relative error: %+.6lf\n", i, finite_diff, relative_error);
    }

    // Clean up
    printf_master("\nCleaning up\n");
    free(grad);
    free(initial_coupling_strength);
    free(new_coupling_strength);
    PetscCall(free_simulation_context(&context));
    PetscCall(SlepcFinalize());
    return 0;
}
