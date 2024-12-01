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
PetscErrorCode run_evolution(Simulation_context *context)
{
    PetscCall(forward_evolution(context));
    PetscScalar dot_product;
    PetscCall(VecDot(context->forward_path[context->time_steps], context->target_vec, &dot_product));
    PetscCall(VecCopy(context->target_vec, context->backward_path[context->time_steps]));
    PetscCall(VecScale(context->backward_path[context->time_steps], dot_product));
    PetscCall(backward_evolution(context));
    return PETSC_SUCCESS;
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
        grad[i] = -2.0 * creal(sum);
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
PetscErrorCode optimize_coupling_strength_gd(Simulation_context *context, int max_iterations, double learning_rate)
{
    printf_master("Start optimizing the coupling strength using gradient descent with ");
    printf_master("max_iterations: %d, learning_rate: %.6e\n", max_iterations, learning_rate);
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
PetscErrorCode optimize_coupling_strength_adam(Simulation_context *context, int max_iterations, double learning_rate, double beta1, double beta2)
{
    printf_master("Start optimizing the coupling strength using Adam optimizer with ");
    printf_master("max_iterations: %d, learning_rate: %.6e, beta1: %.6e, beta2: %.6e\n", max_iterations, learning_rate, beta1, beta2);

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

// Full Adam optimization process with restart
PetscErrorCode optimize_coupling_strength_adam_with_restart(Simulation_context *context, int max_iterations, double learning_rate, double beta1, double beta2)
{
    printf_master("Start optimizing the coupling strength using Adam optimizer with ");
    printf_master("max_iterations: %d, learning_rate: %.6e, beta1: %.6e, beta2: %.6e\n", max_iterations, learning_rate, beta1, beta2);
    double norm2_grad;
    double fidelity;
    double norm2_momentum;
    double *grad = (double *)malloc(context->cnt_bond * sizeof(double));
    double *m = (double *)malloc(context->cnt_bond * sizeof(double));
    double *v = (double *)malloc(context->cnt_bond * sizeof(double));
    int restart_count = 0;
    const int max_restarts = 20;
    const double threshold = 1e-4;
    const int check_after_iters = 20;

    while (restart_count < max_restarts)
    {
        memset(m, 0, context->cnt_bond * sizeof(double));
        memset(v, 0, context->cnt_bond * sizeof(double));
        if (restart_count > 0)
        {
            printf_master("Random initialize coupling strength\n");
            set_random_coupling_strength(context, 1.0, 3.0);
        }

        int iter = 0;
        for (; iter < max_iterations; iter++)
        {
            PetscCall(run_evolution(context));
            PetscCall(calculate_gradient(context, grad));
            norm2_grad = cblas_dnrm2(context->cnt_bond, grad, 1);
            PetscCall(calc_fidelity(context->forward_path[context->time_steps] , context->target_vec, &fidelity));
            PetscCall(write_iteration_data(context, iter, grad, norm2_grad, fidelity));
            print_iteration_data(iter, norm2_grad, fidelity);
            PetscCall(adam_optimizer(context, grad, m, v, beta1, beta2, learning_rate, iter + 1));

            norm2_momentum = cblas_dnrm2(context->cnt_bond, m, 1);

            // Check if stuck in local minimum after some iterations
            if (iter == check_after_iters &&
                norm2_grad < threshold &&
                norm2_momentum < threshold &&
                (1.0 - fidelity) > 0.5)
            {
                printf_master("Optimization stuck, restarting with new random initialization\n");
                break;
            }

            if ((1.0 - fidelity) < 1e-5)
            {
                printf_master("Converged\n");
                free(grad);
                free(m);
                free(v);
                return PETSC_SUCCESS;
            }
        }
        if (iter == max_iterations)
        {
            printf_master("Optimization did not converge\n");
            break;
        }
        restart_count++;
    }

    free(grad);
    free(m);
    free(v);
    return PETSC_SUCCESS;
}

// Random sampling of coupling strength
PetscErrorCode random_sampling_coupling_strength(Simulation_context *context, int cnt_samples, double a, double b)
{
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
