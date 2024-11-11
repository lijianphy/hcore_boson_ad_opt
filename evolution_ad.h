/**
 * @file evolution_ad.h
 * @brief Header file for evolution_ad.c
 */
#ifndef EVOLUTION_AD_H
#define EVOLUTION_AD_H

#include <petscvec.h>
#include <petscmat.h>

#include "hamiltonian.h"

// Function declarations
PetscErrorCode forward_evolution(Simulation_context* context);
PetscErrorCode backward_evolution(Simulation_context* context);
PetscErrorCode run_evolution(Simulation_context* context);
PetscErrorCode calculate_gradient(Simulation_context* context, double* grad);
PetscErrorCode optimize_coupling_strength_gd(Simulation_context *context, int max_iterations, double learning_rate);
PetscErrorCode optimize_coupling_strength_adam(Simulation_context *context, int max_iterations, double learning_rate, double beta1, double beta2);
PetscErrorCode random_initialize_coupling_strength(Simulation_context *context, int max_iterations, double threshold, double *norm2_grad);
PetscErrorCode optimize_coupling_strength_adam_with_restart(Simulation_context *context, int max_iterations, double learning_rate, double beta1, double beta2);
#endif // EVOLUTION_AD_H
