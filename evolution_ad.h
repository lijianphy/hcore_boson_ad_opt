/**
 * @file evolution_ad.h
 * @brief Header file for evolution_ad.c
 */
#ifndef EVOLUTION_AD_H
#define EVOLUTION_AD_H

#include <petscvec.h>
#include <petscmat.h>

#include "hamiltonian.h"

typedef enum
{
    AD_V1 = 1,
    AD_V2 = 2,
    AD_V3 = 3
} AD_TYPE;

// Function declarations
PetscErrorCode forward_evolution(Simulation_context* context);
PetscErrorCode backward_evolution(Simulation_context* context);
PetscErrorCode run_evolution_v1(Simulation_context* context);
PetscErrorCode run_evolution_v2(Simulation_context* context);
PetscErrorCode run_evolution_v3(Simulation_context* context);
PetscErrorCode calculate_gradient(Simulation_context* context, double* grad);
PetscErrorCode optimize_coupling_strength_gd(Simulation_context *context, int max_iterations, double learning_rate, AD_TYPE ad_type);
PetscErrorCode optimize_coupling_strength_adam(Simulation_context *context, int max_iterations, double learning_rate, double beta1, double beta2, AD_TYPE ad_type);
PetscErrorCode random_sampling_coupling_strength(Simulation_context *context, int cnt_samples, double a, double b, AD_TYPE ad_type);
PetscErrorCode optimize_coupling_strength_adam_with_phi(Simulation_context *context, int max_iterations, double learning_rate, double beta1, double beta2);
PetscErrorCode optimize_coupling_strength_adam_parallel(Simulation_context *context_list, int cnt_parallel, PetscScalar *phi_list, int max_iterations, double learning_rate, double beta1, double beta2);
PetscErrorCode optimize_coupling_strength_adam_phi_fix(Simulation_context *context, int max_iterations, double learning_rate, double beta1, double beta2, double phase);
#endif // EVOLUTION_AD_H
