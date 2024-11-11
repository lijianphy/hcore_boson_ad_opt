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

#include "hamiltonian.h"
// #include "evolution_ad.h"
#include "log.h"

// Print the norm of a vector
static PetscErrorCode print_vec_norm(Vec vec, const char *name)
{
    PetscReal norm;
    PetscCall(VecNorm(vec, NORM_2, &norm));
    printf_master("%s norm: %f\n", name, norm);
    return PETSC_SUCCESS;
}

// Print the norm of the difference between two vectors
static PetscErrorCode print_vec_diff_norm(Vec vec1, Vec vec2, const char *name)
{
    Vec diff;
    PetscCall(VecDuplicate(vec1, &diff));
    PetscCall(VecWAXPY(diff, -1.0, vec2, vec1));
    print_vec_norm(diff, name);
    PetscCall(VecDestroy(&diff));
    return PETSC_SUCCESS;
}

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
#ifdef DEBUG
        print_vec_norm(forward_path[k_n], "forward_path");
        print_vec_diff_norm(forward_path[k_n + 1], context->target_vec, "forward_path_diff_target");
        print_vec_diff_norm(forward_path[k_n + 1], context->init_vec, "forward_path_diff_init");
#endif
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
#ifdef DEBUG
        print_vec_norm(backward_path[k_n], "backward_path");
        print_vec_diff_norm(backward_path[k_n - 1], backward_path[k_n], "backward_path_diff");
#endif
    }

    PetscCall(MFNDestroy(&mfn));
    PetscCall(FNDestroy(&f));

    return PETSC_SUCCESS;
}

// Run forward and backward evolution
PetscErrorCode run_evolution(Simulation_context *context)
{
    PetscCall(forward_evolution(context));
    // set initial state for backward evolution as w = forward_path[time_steps] - target_vec
    PetscCall(VecWAXPY(context->backward_path[context->time_steps], -1.0, context->target_vec, context->forward_path[context->time_steps]));
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
        PetscScalar sum = 0;
        Mat m = context->single_bond_hams[i];
        // integrate using trapezoidal rule, see https://en.wikipedia.org/wiki/Trapezoidal_rule
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
        m[i] = beta1 * m[i] + (1 - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i];
        double m_hat = m[i] / (1 - pow(beta1, t));
        double v_hat = v[i] / (1 - pow(beta2, t));
        coupling_strength[i] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
    }
    PetscCall(set_coupling_strength(context, coupling_strength));
    return PETSC_SUCCESS;
}

static PetscErrorCode set_random_coupling_strength(Simulation_context *context)
{
    for (int i = 0; i < context->cnt_bond; i++)
    {
        context->coupling_strength[i] = ((double)rand() / RAND_MAX) * 2.0; // Random value between 0 and 2;
    }
    PetscCall(update_coupling_strength(context, context->coupling_strength));
    return PETSC_SUCCESS;
}

// Randomly initialize the coupling strength such that the gradient is not zero
PetscErrorCode random_initialize_coupling_strength(Simulation_context *context, int max_iterations, double threshold, double *norm2_grad)
{
    double *coupling_strength = context->coupling_strength;
    double *grad = (double *)malloc(context->cnt_bond * sizeof(double));
    double *best_coupling_strength = (double *)malloc(context->cnt_bond * sizeof(double));
    PetscBool grad_is_zero = PETSC_TRUE;
    int iter = 0;
    double norm2_error;
    double max_norm2_grad = 0.0;

    while (grad_is_zero && iter < max_iterations)
    {
        set_random_coupling_strength(context);
        // Run evolution and calculate gradient
        PetscCall(run_evolution(context));
        PetscCall(calculate_gradient(context, grad));

        // Check if gradient is zero
        *norm2_grad = cblas_dnrm2(context->cnt_bond, grad, 1);
        PetscCall(VecNorm(context->backward_path[context->time_steps], NORM_2, &norm2_error));
        printf_master("Random initialization %d: norm2_error: %.6e, norm2_grad: %.6e\n", iter, norm2_error, *norm2_grad);
        if (*norm2_grad > threshold)
        {
            grad_is_zero = PETSC_FALSE;
        }
        if (*norm2_grad > max_norm2_grad)
        {
            max_norm2_grad = *norm2_grad;
            memcpy(best_coupling_strength, coupling_strength, context->cnt_bond * sizeof(double));
        }
        iter++;
    }

    // Set coupling strength to the one with the biggest norm2_grad
    memcpy(coupling_strength, best_coupling_strength, context->cnt_bond * sizeof(double));
    PetscCall(set_coupling_strength(context, coupling_strength));

    free(grad);
    free(best_coupling_strength);
    return PETSC_SUCCESS;
}

// Full gradient descent optimization process
PetscErrorCode optimize_coupling_strength_gd(Simulation_context *context, int max_iterations, double learning_rate)
{
    double norm2_error;
    double norm2_grad;
    double *grad = (double *)malloc(context->cnt_bond * sizeof(double));
    for (int iter = 0; iter < max_iterations; iter++)
    {
        PetscCall(run_evolution(context));
        PetscCall(calculate_gradient(context, grad));
        PetscCall(gradient_descent(context, grad, learning_rate));
        // calculate the norm of diff
        PetscCall(VecNorm(context->backward_path[context->time_steps], NORM_2, &norm2_error));
        // calculate the norm of gradient
        norm2_grad = cblas_dnrm2(context->cnt_bond, grad, 1);
        printf_master("Iteration %d: norm2_error: %.6e, norm2_grad: %.6e\n", iter, norm2_error, norm2_grad);
        if (norm2_error < 1e-5)
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
    double norm2_error;
    double norm2_grad;
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
        // calculate the norm of diff
        PetscCall(VecNorm(context->backward_path[context->time_steps], NORM_2, &norm2_error));
        PetscCall(adam_optimizer(context, grad, m, v, beta1, beta2, learning_rate, iter + 1));
        printf_master("Iteration %4d: norm2_error: %.6e, norm2_grad: %.6e\n", iter, norm2_error, norm2_grad);
        if (norm2_error < 1e-5)
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
    double norm2_error;
    double norm2_grad;
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
            set_random_coupling_strength(context);
        }

        int iter = 0;
        for (; iter < max_iterations; iter++)
        {
            PetscCall(run_evolution(context));
            PetscCall(calculate_gradient(context, grad));
            norm2_grad = cblas_dnrm2(context->cnt_bond, grad, 1);
            PetscCall(VecNorm(context->backward_path[context->time_steps], NORM_2, &norm2_error));
            PetscCall(adam_optimizer(context, grad, m, v, beta1, beta2, learning_rate, iter + 1));

            norm2_momentum = cblas_dnrm2(context->cnt_bond, m, 1);
            printf_master("Iteration %4d (restart %2d): norm2_error: %.6e, norm2_grad: %.6e, norm2_momentum: %.6e\n",
                          iter, restart_count, norm2_error, norm2_grad, norm2_momentum);

            // Check if stuck in local minimum after some iterations
            if (iter == check_after_iters &&
                norm2_grad < threshold &&
                norm2_momentum < threshold &&
                norm2_error > 0.5)
            {
                printf_master("Optimization stuck, restarting with new random initialization\n");
                break;
            }

            if (norm2_error < 1e-5)
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
