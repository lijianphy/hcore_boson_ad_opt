/**
 * @file evolution_ad.c
 * @brief Implementation of automatic differentiation (AD) for real-time evolution
 *        in optimizing the coupling strength of a quantum system.
 *
 * This file contains functions for performing forward and backward evolution
 * of a quantum system, calculating gradients, and optimizing coupling strengths
 * using gradient descent and the Adam optimizer. The main functionalities include:
 * - Forward and backward evolution of the system state vectors.
 * - Calculation of gradients with respect to the coupling strengths.
 * - Optimization of coupling strengths using gradient descent and Adam optimizer.
 * - Random initialization of coupling strengths to ensure non-zero gradients.
 */

#include <petscvec.h>
#include <petscmat.h>
#include <slepcfn.h>
#include <slepcmfn.h>
#include <math.h>
#include <complex.h>
#include <cblas.h>
#include <time.h>
#include <mpi.h>

#include "hamiltonian.h"
#include "evolution_ad.h"
#include "log.h"
#include "random.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Run forward evolution
PetscErrorCode forward_evolution(Simulation_context *context)
{
    double delta_t = context->total_time / context->time_steps;
    Mat ham = context->hamiltonian;
    Vec *forward_path = context->forward_path;

    FN f;
    PetscCall(FNCreate(PETSC_COMM_WORLD, &f));
    PetscCall(FNSetType(f, FNEXP));
    PetscCall(FNSetScale(f, -I * delta_t, 1.0));

    MFN mfn;
    PetscCall(MFNCreate(PETSC_COMM_WORLD, &mfn));
    PetscCall(MFNSetFN(mfn, f));
    PetscCall(MFNSetType(mfn, MFNEXPOKIT));
    PetscCall(MFNSetOperator(mfn, ham));
    PetscCall(MFNSetFromOptions(mfn));

    for (int k_n = 0; k_n < context->time_steps; k_n++)
    {
        // printf_master("Forward evolution step %d\n", k_n);
        PetscCall(MFNSolve(mfn, forward_path[k_n], forward_path[k_n + 1]));
    }

    PetscCall(MFNDestroy(&mfn));
    PetscCall(FNDestroy(&f));

    return PETSC_SUCCESS;
}

// Run backward evolution
PetscErrorCode backward_evolution(Simulation_context *context)
{
    double delta_t = context->total_time / context->time_steps;
    Mat ham = context->hamiltonian;
    Vec *backward_path = context->backward_path;

    FN f;
    PetscCall(FNCreate(PETSC_COMM_WORLD, &f));
    PetscCall(FNSetType(f, FNEXP));
    PetscCall(FNSetScale(f, I * delta_t, 1.0));

    MFN mfn;
    PetscCall(MFNCreate(PETSC_COMM_WORLD, &mfn));
    PetscCall(MFNSetFN(mfn, f));
    PetscCall(MFNSetType(mfn, MFNEXPOKIT));
    PetscCall(MFNSetOperator(mfn, ham));
    PetscCall(MFNSetFromOptions(mfn));

    for (int k_n = context->time_steps; k_n > 0; k_n--)
    {
        // printf_master("Backward evolution step %d\n", k_n);
        PetscCall(MFNSolve(mfn, backward_path[k_n], backward_path[k_n - 1]));
    }

    PetscCall(MFNDestroy(&mfn));
    PetscCall(FNDestroy(&f));

    return PETSC_SUCCESS;
}

// Run forward and backward evolution
PetscErrorCode run_evolution_v1(Simulation_context *context)
{
    PetscCall(forward_evolution(context));
    // set initial state for backward evolution as w = forward_path[time_steps] - target_vec
    PetscCall(VecWAXPY(context->backward_path[context->time_steps], -1.0, context->target_vec, context->forward_path[context->time_steps]));
    PetscCall(backward_evolution(context));
    return PETSC_SUCCESS;
}

// Run forward and backward evolution
PetscErrorCode run_evolution_v2(Simulation_context *context)
{
    PetscCall(forward_evolution(context));
    PetscScalar dot_product;
    PetscCall(VecDot(context->forward_path[context->time_steps], context->target_vec, &dot_product));
    PetscCall(VecCopy(context->target_vec, context->backward_path[context->time_steps]));
    PetscCall(VecScale(context->backward_path[context->time_steps], -1.0 * dot_product));
    PetscCall(backward_evolution(context));
    return PETSC_SUCCESS;
}

// Run forward and backward evolution
PetscErrorCode run_evolution_v3(Simulation_context *context)
{
    PetscCall(forward_evolution(context));
    PetscScalar dot_product;
    PetscCall(VecDot(context->forward_path[context->time_steps], context->target_vec, &dot_product));
    double norm = cabs(dot_product);
    if (norm < 1e-10)
    {
        dot_product = 1.0;
    } else {
        dot_product /= norm;
    }
    PetscCall(VecWAXPY(context->backward_path[context->time_steps], -1.0 * dot_product, context->target_vec, context->forward_path[context->time_steps]));
    PetscCall(backward_evolution(context));
    return PETSC_SUCCESS;
}

// Run forward and backward evolution
PetscErrorCode run_evolution_with_phi(Simulation_context *context, PetscScalar phi)
{
    PetscCall(forward_evolution(context));
    PetscScalar phi_normalized = phi / cabs(phi);
    PetscCall(VecWAXPY(context->backward_path[context->time_steps], -1.0 * phi_normalized, context->target_vec, context->forward_path[context->time_steps]));
    PetscCall(backward_evolution(context));
    return PETSC_SUCCESS;
}

// Add this helper function before all optimization functions
static PetscErrorCode (*get_evolution_function(AD_TYPE ad_type))(Simulation_context*)
{
    switch (ad_type)
    {
    case AD_V1:
        return run_evolution_v1;
    case AD_V2:
        return run_evolution_v2;
    case AD_V3:
        return run_evolution_v3;
    default:
        return NULL;
    }
}

// Calculate the gradient
PetscErrorCode calculate_gradient(Simulation_context *context, double *grad)
{
    Vec *forward_path = context->forward_path;
    Vec *backward_path = context->backward_path;
    Vec init_vec = context->init_vec;
    double delta_t = context->total_time / context->time_steps;
    int time_steps = context->time_steps;

    Vec tmp;
    PetscCall(VecDuplicate(init_vec, &tmp));
    for (int i = 0; i < context->cnt_bond; i++) // for each bond
    {
        if (context->isfixed[i]) // skip fixed bonds
        {
            grad[i] = 0.0;
            continue;
        }
        PetscScalar sum = 0.0;
        Mat m = context->single_bond_hams[i];
        // integrate using Trapezoidal rule, see https://en.wikipedia.org/wiki/Trapezoidal_rule
        for (int t = 0; t <= time_steps; t++)
        {
            PetscScalar result;
            Vec f = forward_path[t];
            Vec b = backward_path[t];
            PetscCall(MatMult(m, f, tmp));
            PetscCall(VecDot(tmp, b, &result));
            double factor = (t == 0 || t == time_steps) ? 0.5 : 1.0;
            sum += result * delta_t * factor;
        }
        grad[i] = 2.0 * creal(sum);
    }
    PetscCall(VecDestroy(&tmp));

    return PETSC_SUCCESS;
}

// update the coupling strength using simple gradient descent
// see https://en.wikipedia.org/wiki/Gradient_descent
static PetscErrorCode gradient_descent(Simulation_context *context, double *grad, double learning_rate)
{
    double *coupling_strength = context->coupling_strength;
    for (int i = 0; i < context->cnt_bond; i++)
    {
        if (context->isfixed[i])
        {
            continue;
        }
        coupling_strength[i] -= learning_rate * grad[i];
    }
    PetscCall(set_coupling_strength(context, coupling_strength));
    return PETSC_SUCCESS;
}

// update the coupling strength using the Adam optimizer
// see https://arxiv.org/abs/1412.6980
static PetscErrorCode adam_optimizer(Simulation_context *context, double *grad, double *m, double *v, double beta1, double beta2, double learning_rate, int t)
{
    double *coupling_strength = context->coupling_strength;
    double epsilon = 1e-8;
    for (int i = 0; i < context->cnt_bond; i++)
    {
        if (context->isfixed[i])
        {
            continue;
        }
        m[i] = beta1 * m[i] + (1 - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i];
        double m_hat = m[i] / (1 - pow(beta1, t));
        double v_hat = v[i] / (1 - pow(beta2, t));
        coupling_strength[i] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
    }
    PetscCall(set_coupling_strength(context, coupling_strength));
    return PETSC_SUCCESS;
}

// Set the coupling strength to random values
static PetscErrorCode set_random_coupling_strength(Simulation_context *context, double a, double b)
{
    // Only master process generates random values
    if (context->partition_id == 0) {
        for (int i = 0; i < context->cnt_bond; i++) {
            if (!context->isfixed[i]) {
                context->coupling_strength[i] = randu2(context->rng, a, b);
            }
        }
    }
    
    // Set Hamiltonian with the new coupling strengths
    PetscCall(set_coupling_strength(context, context->coupling_strength));
    return PETSC_SUCCESS;
}

// Write iteration data before the optimization functions (which will update the coupling strength)
static PetscErrorCode write_iteration_data(Simulation_context *context, int iter, double *grad, double norm2_grad, double fidelity)
{
    FILE *fp = context->output_file;
    if (fp && context->partition_id == 0)
    {
        fprintf(fp, "{\"iteration\": %d, \"coupling_strength\": [", iter);
        for (int i = 0; i < context->cnt_bond; i++)
        {
            fprintf(fp, "%.10e%s", context->coupling_strength[i],
                    i < context->cnt_bond - 1 ? ", " : "");
        }
        fprintf(fp, "], \"gradient\": [");
        for (int i = 0; i < context->cnt_bond; i++)
        {
            fprintf(fp, "%.10e%s", grad[i],
                    i < context->cnt_bond - 1 ? ", " : "");
        }
        fprintf(fp, "], \"norm2_grad\": %.10e, \"infidelity\": %.10e}\n",
                norm2_grad, 1.0 - fidelity);
        fflush(fp);
    }
    return PETSC_SUCCESS;
}

// Print iteration data during the optimization process
static void print_iteration_data(int iter, double norm2_grad, double fidelity)
{
    time_t now;
    char timestamp[20];
    time(&now);
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&now));

    printf_master("[%s] Iteration %4d: infidelity: %.6e, norm2_grad: %.6e\n", timestamp, iter, 1.0 - fidelity, norm2_grad);
}

// Full gradient descent optimization process
PetscErrorCode optimize_coupling_strength_gd(Simulation_context *context, int max_iterations, double learning_rate, AD_TYPE ad_type)
{
    printf_master("Start optimizing the coupling strength using gradient descent with ");
    printf_master("max_iterations: %d, learning_rate: %.6e\n", max_iterations, learning_rate);
    
    PetscErrorCode (*run_evolution)(Simulation_context*) = get_evolution_function(ad_type);
    PetscCheck(run_evolution, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid AD type");

    double norm2_grad;
    double fidelity;
    double *grad = (double *)malloc(context->cnt_bond * sizeof(double));
    for (int iter = 0; iter < max_iterations; iter++)
    {
        PetscCall(run_evolution(context));
        PetscCall(calculate_gradient(context, grad));
        PetscCall(calc_fidelity(context->forward_path[context->time_steps] , context->target_vec, &fidelity));
        // calculate the norm of gradient
        norm2_grad = cblas_dnrm2(context->cnt_bond, grad, 1);
        PetscCall(write_iteration_data(context, iter, grad, norm2_grad, fidelity));
        print_iteration_data(iter, norm2_grad, fidelity);
        PetscCall(gradient_descent(context, grad, learning_rate));
        if (1.0 - fidelity < 1e-5)
        {
            printf_master("Converged\n");
            break;
        }
    }
    free(grad);
    return PETSC_SUCCESS;
}

// Full Adam optimization process
PetscErrorCode optimize_coupling_strength_adam(Simulation_context *context, int max_iterations, double learning_rate, double beta1, double beta2, AD_TYPE ad_type)
{
    printf_master("Start optimizing the coupling strength using Adam optimizer with ");
    printf_master("max_iterations: %d, learning_rate: %.6e, beta1: %.6e, beta2: %.6e\n", max_iterations, learning_rate, beta1, beta2);
    
    PetscErrorCode (*run_evolution)(Simulation_context*) = get_evolution_function(ad_type);
    PetscCheck(run_evolution, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid AD type");

    double norm2_grad;
    double fidelity;
    double *grad = (double *)malloc(context->cnt_bond * sizeof(double));
    double *m = (double *)malloc(context->cnt_bond * sizeof(double));
    double *v = (double *)malloc(context->cnt_bond * sizeof(double));
    memset(m, 0, context->cnt_bond * sizeof(double));
    memset(v, 0, context->cnt_bond * sizeof(double));

    for (int iter = 0; iter < max_iterations; iter++)
    {
        PetscCall(run_evolution(context));
        PetscCall(calculate_gradient(context, grad));
        // calculate the norm of gradient
        norm2_grad = cblas_dnrm2(context->cnt_bond, grad, 1);
        PetscCall(calc_fidelity(context->forward_path[context->time_steps] , context->target_vec, &fidelity));
        PetscCall(write_iteration_data(context, iter, grad, norm2_grad, fidelity));
        print_iteration_data(iter, norm2_grad, fidelity);
        PetscCall(adam_optimizer(context, grad, m, v, beta1, beta2, learning_rate, iter % 100 + 1));

        if (1.0 - fidelity < 1e-5)
        {
            printf_master("Converged\n");
            break;
        }
    }
    free(grad);
    free(m);
    free(v);
    return PETSC_SUCCESS;
}

PetscErrorCode get_random_phi(Simulation_context *context, PetscScalar *phi)
{
    if (context->partition_id == 0) {
        double phase = randu2(context->rng, 0.0, 2.0 * M_PI);
        PetscScalar phi_real = cos(phase);
        PetscScalar phi_imag = sin(phase);
        *phi = phi_real + I * phi_imag;
    }
    // broadcast phi to all processes
    PetscCall(MPI_Bcast(phi, 2, MPI_DOUBLE, 0, PETSC_COMM_WORLD));

    return PETSC_SUCCESS;
}

// Full Adam optimization process with changing phi
PetscErrorCode optimize_coupling_strength_adam_with_phi(Simulation_context *context, int max_iterations, double learning_rate, double beta1, double beta2)
{
    printf_master("Start optimizing the coupling strength using Adam optimizer with ");
    printf_master("max_iterations: %d, learning_rate: %.6e, beta1: %.6e, beta2: %.6e\n", max_iterations, learning_rate, beta1, beta2);

    PetscScalar phi = 1.0;
    double norm2_grad;
    double fidelity;
    double infidelity;
    double *grad = (double *)malloc(context->cnt_bond * sizeof(double));
    double *m = (double *)malloc(context->cnt_bond * sizeof(double));
    double *v = (double *)malloc(context->cnt_bond * sizeof(double));
    memset(m, 0, context->cnt_bond * sizeof(double));
    memset(v, 0, context->cnt_bond * sizeof(double));

    int phi_change_cooldown = 0; // Prevent too frequent phi changes

    for (int iter = 0; iter < max_iterations; iter++)
    {
        PetscCall(run_evolution_with_phi(context, phi));
        PetscCall(calculate_gradient(context, grad));
        norm2_grad = cblas_dnrm2(context->cnt_bond, grad, 1);
        PetscCall(calc_fidelity(context->forward_path[context->time_steps], context->target_vec, &fidelity));
        infidelity = 1.0 - fidelity;

        // Check if we should change phi
        if (phi_change_cooldown >= 10 && norm2_grad < 1e-3 && infidelity > 0.1) {
            printf_master("Gradient too small (%.2e), generating new phi\n", norm2_grad);
            PetscCall(get_random_phi(context, &phi));
            phi_change_cooldown = 0;
        }
        phi_change_cooldown++;

        PetscCall(write_iteration_data(context, iter, grad, norm2_grad, fidelity));
        print_iteration_data(iter, norm2_grad, fidelity);
        PetscCall(adam_optimizer(context, grad, m, v, beta1, beta2, learning_rate, iter % 100 + 1));

        if (infidelity < 1e-5)
        {
            printf_master("Converged\n");
            break;
        }
    }
    free(grad);
    free(m);
    free(v);
    return PETSC_SUCCESS;
}

// Random sampling of coupling strength
PetscErrorCode random_sampling_coupling_strength(Simulation_context *context, int cnt_samples, double a, double b, AD_TYPE ad_type)
{
    PetscErrorCode (*run_evolution)(Simulation_context*) = get_evolution_function(ad_type);
    PetscCheck(run_evolution, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid AD type");

    double *grad = (double *)malloc(context->cnt_bond * sizeof(double));
    double norm2_grad;
    double fidelity;
    for (int i = 0; i < cnt_samples; i++)
    {
        set_random_coupling_strength(context, a, b);
        PetscCall(run_evolution(context));
        PetscCall(calculate_gradient(context, grad));
        // calculate the norm of gradient
        norm2_grad = cblas_dnrm2(context->cnt_bond, grad, 1);
        PetscCall(calc_fidelity(context->forward_path[context->time_steps] , context->target_vec, &fidelity));
        PetscCall(write_iteration_data(context, i, grad, norm2_grad, fidelity));
        print_iteration_data(i, norm2_grad, fidelity);
    }

    free(grad);
    return PETSC_SUCCESS;
}
