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
    PetscCall(FNCreate(context->comm, &f));
    PetscCall(FNSetType(f, FNEXP));
    PetscCall(FNSetScale(f, -I * delta_t, 1.0));

    MFN mfn;
    PetscCall(MFNCreate(context->comm, &mfn));
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
    PetscCall(FNCreate(context->comm, &f));
    PetscCall(FNSetType(f, FNEXP));
    PetscCall(FNSetScale(f, I * delta_t, 1.0));

    MFN mfn;
    PetscCall(MFNCreate(context->comm, &mfn));
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

// Run forward and backward evolution, version 1
// loss = |forward_path[time_steps] - target_vec|^2
PetscErrorCode run_evolution_v1(Simulation_context *context)
{
    PetscCall(forward_evolution(context));
    // set initial state for backward evolution as w = forward_path[time_steps] - target_vec
    PetscCall(VecWAXPY(context->backward_path[context->time_steps], -1.0, context->target_vec, context->forward_path[context->time_steps]));
    PetscCall(backward_evolution(context));
    return PETSC_SUCCESS;
}

// Run forward and backward evolution, version 2
// loss = 1.0 - |<forward_path[time_steps] | target_vec>|^2
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

// Run forward and backward evolution, version 3
// loss = |forward_path[time_steps] - |target_vec><target_vec | forward_path[time_steps]> |^2
PetscErrorCode run_evolution_v3(Simulation_context *context)
{
    PetscCall(forward_evolution(context));
    PetscScalar dot_product;
    PetscCall(VecDot(context->forward_path[context->time_steps], context->target_vec, &dot_product));
    double norm = cabs(dot_product);
    if (norm < 1e-10)
    {
        dot_product = 1.0;
    }
    else
    {
        dot_product /= norm;
    }
    PetscCall(VecWAXPY(context->backward_path[context->time_steps], -1.0 * dot_product, context->target_vec, context->forward_path[context->time_steps]));
    PetscCall(backward_evolution(context));
    return PETSC_SUCCESS;
}

// Run forward and backward evolution, version 4
// loss = log(1.0 - |<forward_path[time_steps] | target_vec>|^2)
PetscErrorCode run_evolution_v4(Simulation_context *context)
{
    PetscCall(forward_evolution(context));
    PetscScalar dot_product;
    PetscCall(VecDot(context->forward_path[context->time_steps], context->target_vec, &dot_product));
    double loss_inv = 1.0 / (1.0 - cabs(dot_product) * cabs(dot_product));
    PetscCall(VecCopy(context->target_vec, context->backward_path[context->time_steps]));
    PetscCall(VecScale(context->backward_path[context->time_steps], -loss_inv * dot_product));
    PetscCall(backward_evolution(context));
    return PETSC_SUCCESS;
}

// Run forward and backward evolution with a phase factor
// loss = |forward_path[time_steps] - phase_factor * target_vec|^2
PetscErrorCode run_evolution_with_phase(Simulation_context *context, double phi)
{
    PetscCall(forward_evolution(context));
    PetscScalar phase_factor = cos(phi) + I * sin(phi);
    PetscCall(VecWAXPY(context->backward_path[context->time_steps], -1.0 * phase_factor, context->target_vec, context->forward_path[context->time_steps]));
    PetscCall(backward_evolution(context));
    return PETSC_SUCCESS;
}

// Add this helper function before all optimization functions
static PetscErrorCode (*get_evolution_function(AD_TYPE ad_type))(Simulation_context *)
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
    if (context->partition_id == 0)
    {
        for (int i = 0; i < context->cnt_bond; i++)
        {
            if (!context->isfixed[i])
            {
                context->coupling_strength[i] = randu2(context->rng, a, b);
            }
        }
    }

    // Set Hamiltonian with the new coupling strengths
    PetscCall(set_coupling_strength(context, context->coupling_strength));
    return PETSC_SUCCESS;
}

// Set the coupling strength to random values with normal distribution
static PetscErrorCode set_random_coupling_strength_normal(Simulation_context *context, double mu, double sigma)
{
    // Only master process generates random values
    if (context->partition_id == 0)
    {
        for (int i = 0; i < context->cnt_bond; i++)
        {
            if (!context->isfixed[i])
            {
                context->coupling_strength[i] = randn2(context->rng, mu, sigma);
            }
        }
    }

    // Set Hamiltonian with the new coupling strengths
    PetscCall(set_coupling_strength(context, context->coupling_strength));
    return PETSC_SUCCESS;
}

// Write iteration data before the optimization functions (which will update the coupling strength)
static PetscErrorCode write_iteration_data(Simulation_context *context, int iter, double norm2_grad, double fidelity)
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
        fprintf(fp, "], \"norm2_grad\": %.10e, \"infidelity\": %.10e}\n",
                norm2_grad, 1.0 - fidelity);
        fflush(fp);
    }
    return PETSC_SUCCESS;
}

// Print iteration data during the optimization process
static void print_iteration_data(int iter, double norm2_grad, double fidelity, int stream_id, int partition_id)
{
    if (partition_id != 0)
    {
        return;
    }
    printf("[%2d] Iteration %4d: infidelity: %.6e, norm2_grad: %.6e\n", stream_id, iter, 1.0 - fidelity, norm2_grad);
}

// Full gradient descent optimization process
PetscErrorCode optimize_coupling_strength_gd(Simulation_context *context, int max_iterations, double learning_rate, AD_TYPE ad_type)
{
    printf_master("Start optimizing the coupling strength using gradient descent with\n");
    printf_master("max_iterations: %d, learning_rate: %.6e\n", max_iterations, learning_rate);

    PetscErrorCode (*run_evolution)(Simulation_context *) = get_evolution_function(ad_type);
    PetscCheck(run_evolution, context->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid AD type");

    double norm2_grad;
    double fidelity;
    double *grad = (double *)malloc(context->cnt_bond * sizeof(double));
    for (int iter = 0; iter < max_iterations; iter++)
    {
        PetscCall(run_evolution(context));
        PetscCall(calculate_gradient(context, grad));
        PetscCall(calc_fidelity(context->forward_path[context->time_steps], context->target_vec, &fidelity));
        // calculate the norm of gradient
        norm2_grad = cblas_dnrm2(context->cnt_bond, grad, 1);
        PetscCall(write_iteration_data(context, iter, norm2_grad, fidelity));
        print_iteration_data(iter, norm2_grad, fidelity, context->stream_id, context->partition_id);
        if ((1.0 - fidelity) < 1e-5)
        {
            PetscPrintf(context->comm, "Stream %d Converged\n", context->stream_id);
            break;
        }
        PetscCall(gradient_descent(context, grad, learning_rate));
    }
    free(grad);
    return PETSC_SUCCESS;
}

// Full Adam optimization process
PetscErrorCode optimize_coupling_strength_adam(Simulation_context *context, int max_iterations, double learning_rate, double beta1, double beta2, AD_TYPE ad_type)
{
    printf_master("Start optimizing the coupling strength using Adam optimizer with\n");
    printf_master("max_iterations: %d, learning_rate: %.6e, beta1: %.6e, beta2: %.6e\n", max_iterations, learning_rate, beta1, beta2);

    PetscErrorCode (*run_evolution)(Simulation_context *) = get_evolution_function(ad_type);
    PetscCheck(run_evolution, context->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid AD type");

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
        PetscCall(calc_fidelity(context->forward_path[context->time_steps], context->target_vec, &fidelity));
        PetscCall(write_iteration_data(context, iter, norm2_grad, fidelity));
        print_iteration_data(iter, norm2_grad, fidelity, context->stream_id, context->partition_id);

        if ((1.0 - fidelity) < 1e-5)
        {
            PetscPrintf(context->comm, "Stream %d Converged\n", context->stream_id);
            break;
        }

        PetscCall(adam_optimizer(context, grad, m, v, beta1, beta2, learning_rate, iter % 100 + 1));
    }
    free(grad);
    free(m);
    free(v);
    return PETSC_SUCCESS;
}

// get random phase from [0, 2*pi)
static PetscErrorCode get_random_phi(Simulation_context *context, double *phi)
{
    if (context->partition_id == 0)
    {
        *phi = randu2(context->rng, 0.0, 2.0 * M_PI);
    }
    // broadcast phi to all processes
    PetscCall(MPI_Bcast(phi, 1, MPI_DOUBLE, 0, context->comm));
    return PETSC_SUCCESS;
}

// get phi by adding a fixed value (0.1) to the previous phi
PetscErrorCode get_random_phi_v2(Simulation_context *context, double *phi)
{
    if (context->partition_id == 0)
    {
        *phi += 0.1;
    }
    // broadcast phi to all processes
    PetscCall(MPI_Bcast(phi, 1, MPI_DOUBLE, 0, context->comm));

    return PETSC_SUCCESS;
}

// get phi by adding a fixed value (0.1 + pi/2) to the previous phi
PetscErrorCode get_random_phi_v3(Simulation_context *context, double *phi)
{
    if (context->partition_id == 0)
    {
        *phi += 0.1 + M_PI / 2.0;
    }
    // broadcast phi to all processes
    PetscCall(MPI_Bcast(phi, 1, MPI_DOUBLE, 0, context->comm));

    return PETSC_SUCCESS;
}

// Full Adam optimization process with fixed phase factor
PetscErrorCode optimize_coupling_strength_adam_fixed_phase(Simulation_context *context, int max_iterations, double learning_rate, double beta1, double beta2, double phi)
{
    printf_master("Start optimizing the coupling strength using Adam optimizer with\n");
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
        PetscCall(run_evolution_with_phase(context, phi));
        PetscCall(calculate_gradient(context, grad));
        // calculate the norm of gradient
        norm2_grad = cblas_dnrm2(context->cnt_bond, grad, 1);
        PetscCall(calc_fidelity(context->forward_path[context->time_steps], context->target_vec, &fidelity));
        PetscCall(write_iteration_data(context, iter, norm2_grad, fidelity));
        print_iteration_data(iter, norm2_grad, fidelity, context->stream_id, context->partition_id);

        if ((1.0 - fidelity) < 1e-5)
        {
            PetscPrintf(context->comm, "Stream %d Converged\n", context->stream_id);
            break;
        }

        PetscCall(adam_optimizer(context, grad, m, v, beta1, beta2, learning_rate, iter % 100 + 1));
    }
    free(grad);
    free(m);
    free(v);
    return PETSC_SUCCESS;
}

// Full Adam optimization process with changing phase factor
PetscErrorCode optimize_coupling_strength_adam_changing_phase(Simulation_context *context, int max_iterations, double learning_rate, double beta1, double beta2, int type)
{
    printf_master("Start optimizing the coupling strength using Adam optimizer with\n");
    printf_master("max_iterations: %d, learning_rate: %.6e, beta1: %.6e, beta2: %.6e\n", max_iterations, learning_rate, beta1, beta2);

    double phi = 0.0;
    double norm2_grad;
    double fidelity;
    double infidelity;
    double *grad = (double *)malloc(context->cnt_bond * sizeof(double));
    double *m = (double *)malloc(context->cnt_bond * sizeof(double));
    double *v = (double *)malloc(context->cnt_bond * sizeof(double));
    memset(m, 0, context->cnt_bond * sizeof(double));
    memset(v, 0, context->cnt_bond * sizeof(double));

    // Buffer to store the last 10 infidelities
    const int buffer_size = 10;
    double *infidelity_buffer = (double *)malloc(buffer_size * sizeof(double));
    memset(infidelity_buffer, 0, buffer_size * sizeof(double));
    int buffer_idx = 0;
    int change_cooldown = 0;
    int adam_iter = 0;

    for (int iter = 0; iter < max_iterations; iter++)
    {
        PetscCall(run_evolution_with_phase(context, phi));
        PetscCall(calculate_gradient(context, grad));
        norm2_grad = cblas_dnrm2(context->cnt_bond, grad, 1);
        PetscCall(calc_fidelity(context->forward_path[context->time_steps], context->target_vec, &fidelity));
        infidelity = 1.0 - fidelity;

        // Store current infidelity in circular buffer
        infidelity_buffer[buffer_idx] = infidelity;
        buffer_idx = (buffer_idx + 1) % buffer_size;

        PetscCall(write_iteration_data(context, iter, norm2_grad, fidelity));
        print_iteration_data(iter, norm2_grad, fidelity, context->stream_id, context->partition_id);

        if (infidelity < 1e-5)
        {
            PetscPrintf(context->comm, "Stream %d Converged\n", context->stream_id);
            break;
        }

        // Calculate average change rate
        if (change_cooldown > 20)
        {
            double avg_change_rate = (infidelity_buffer[buffer_idx] - infidelity_buffer[(buffer_idx + buffer_size - 1) % buffer_size]) / buffer_size;
            avg_change_rate /= infidelity;

            if (fabs(avg_change_rate) < 1e-6 && infidelity > 0.01)
            {
                switch (type)
                {
                case 1:
                    PetscCall(get_random_phi(context, &phi));
                    break;
                case 2:
                    PetscCall(get_random_phi_v2(context, &phi));
                    break;
                case 3:
                    PetscCall(get_random_phi_v3(context, &phi));
                    break;
                default:
                    PetscCall(get_random_phi(context, &phi));
                    break;
                }
                PetscPrintf(context->comm, "Stream %d: Average change rate too small (%.2e), generating new phi (%.6f)\n", context->stream_id, avg_change_rate, phi);
                change_cooldown = 0;
                buffer_idx = 0;
                memset(m, 0, context->cnt_bond * sizeof(double));
                memset(v, 0, context->cnt_bond * sizeof(double));
                adam_iter = 0;
                continue;
            }
        }
        change_cooldown++;
        PetscCall(adam_optimizer(context, grad, m, v, beta1, beta2, learning_rate, adam_iter % 100 + 1));
        adam_iter++;
    }

    free(grad);
    free(m);
    free(v);
    free(infidelity_buffer);
    return PETSC_SUCCESS;
}

// sort the index list based on fidelities in descending order
void sort_index_by_fidelity(const double *fidelities, int *index_list, int cnt_stream)
{
    // do the sort using insertion sort
    for (int i = 1; i < cnt_stream; i++)
    {
        int index = index_list[i];
        double fidelity = fidelities[index];
        int j = i - 1;
        while (j >= 0 && fidelities[index_list[j]] < fidelity)
        {
            index_list[j + 1] = index_list[j];
            j--;
        }
        index_list[j + 1] = index;
    }
}

// Random sampling of coupling strength
PetscErrorCode random_sampling_coupling_strength(Simulation_context *context, int cnt_samples, double a, double b, AD_TYPE ad_type)
{
    PetscErrorCode (*run_evolution)(Simulation_context *) = get_evolution_function(ad_type);
    PetscCheck(run_evolution, context->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid AD type");

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
        PetscCall(calc_fidelity(context->forward_path[context->time_steps], context->target_vec, &fidelity));
        PetscCall(write_iteration_data(context, i, norm2_grad, fidelity));
        print_iteration_data(i, norm2_grad, fidelity, context->stream_id, context->partition_id);
    }

    free(grad);
    return PETSC_SUCCESS;
}

static inline int min_int(int a, int b)
{
    return a < b ? a : b;
}

static inline double max_double(double a, double b)
{
    return a > b ? a : b;
}

static inline double min_double(double a, double b)
{
    return a < b ? a : b;
}

// Learning rate schedule using CosineAnnealing
static double learning_rate_schedule(int iter, int max_iterations, double lr_min, double lr_max)
{
    double lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(M_PI * iter / max_iterations));
    return lr;
}

// slope of linear regression
static double linear_slope(const double *y, int n, int p)
{
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_xy = 0.0;
    double sum_x2 = 0.0;
    for (int i = 0; i < n; i++)
    {
        int j = (p + i) % n;
        sum_x += i;
        sum_y += y[j];
        sum_xy += i * y[j];
        sum_x2 += i * i;
    }
    double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    return slope;
}

// Full Adam optimization process with parallel instances
PetscErrorCode optimize_coupling_strength_adam_parallel(Simulation_context *context, double learning_rate, double beta1, double beta2)
{
    int cnt_parallel = context->n_streams;
    int max_iterations = context->total_iteration;
    printf_master("Start optimizing the coupling strength using parallel Adam optimizer with\n");
    printf_master("    parallel size: %d, max_iterations: %d, learning_rate: %.6e, beta1: %.6e, beta2: %.6e\n",
                  cnt_parallel, max_iterations, learning_rate, beta1, beta2);

    if (context->stream_id != 0)
    {
        // PetscCall(set_random_coupling_strength_normal(context, 5.0, 2.0));
        PetscCall(set_random_coupling_strength(context, 0.5, 10.0));
    }

    int total_rank_id;
    MPI_Comm_rank(PETSC_COMM_WORLD, &total_rank_id);

    int cnt_bond = context->cnt_bond;
    double norm2_grad;
    double fidelity;
    double infidelity;

    // Allocate memory for optimization variables
    double *grad = (double *)malloc(cnt_bond * sizeof(double));
    double *m = (double *)malloc(cnt_bond * sizeof(double));
    double *v = (double *)malloc(cnt_bond * sizeof(double));
    memset(m, 0, cnt_bond * sizeof(double));
    memset(v, 0, cnt_bond * sizeof(double));

    // Buffer to store the last 10 infidelities
    const int buffer_size = 10;
    const int change_cooldown_threshold = 50;
    double *infidelity_buffer = (double *)malloc(buffer_size * sizeof(double));
    memset(infidelity_buffer, 0, buffer_size * sizeof(double));
    int buffer_idx = 0;
    int change_cooldown = 0;
    int adam_iter = 0;
    int converged = 0;
    int converged_any = 0;
    double max_fidelity = 0.0;
    double max_fidelity_change_rate = 1e3;

    double *fidelities = NULL;
    double *change_rates = NULL;
    int *index_list = NULL;
    double *coupling_strength_list = NULL;
    int *is_coupling_reset_list = NULL;
    if (context->root_id == total_rank_id)
    {
        fidelities = (double *)malloc(cnt_parallel * sizeof(double));
        index_list = (int *)malloc(cnt_parallel * sizeof(int));
        coupling_strength_list = (double *)malloc(cnt_parallel * cnt_bond * sizeof(double));
        for (int i = 0; i < cnt_parallel; i++)
        {
            index_list[i] = i;
        }
        is_coupling_reset_list = (int *)calloc(cnt_parallel, sizeof(int));
        change_rates = (double *)malloc(cnt_parallel * sizeof(double));
    }

    int is_coupling_reset = 0;
    int is_coupling_reset_any = 0;
    double avg_change_rate = 1e3;

    double *coupling_strength_reset = (double *)malloc(cnt_bond * sizeof(double));
    memcpy(coupling_strength_reset, context->coupling_strength, cnt_bond * sizeof(double));

    for (int iter = 0; iter < max_iterations; iter++)
    {
        // Run one iteration for each parallel stream
        PetscCall(run_evolution_v4(context));
        PetscCall(calculate_gradient(context, grad));
        norm2_grad = cblas_dnrm2(cnt_bond, grad, 1);
        PetscCall(calc_fidelity(context->forward_path[context->time_steps], context->target_vec, &fidelity));
        infidelity = 1.0 - fidelity;

        // Store current infidelity in circular buffer
        infidelity_buffer[buffer_idx] = infidelity;
        buffer_idx = (buffer_idx + 1) % buffer_size;

        PetscCall(write_iteration_data(context, iter, norm2_grad, fidelity));
        // print_iteration_data(iter, norm2_grad, fidelity, context->stream_id, context->partition_id);

        if ((1.0 - fidelity) < 1e-3)
        {
            PetscPrintf(context->comm, "[%5d] Stream %d Converged\n", iter, context->stream_id);
            converged = 1;
        }

        // Check if any stream has converged
        PetscCall(MPI_Allreduce(&converged, &converged_any, 1, MPI_INT, MPI_LOR, PETSC_COMM_WORLD));
        if (converged_any)
        {
            break;
        }

        // caclulate change rate
        if (change_cooldown > change_cooldown_threshold)
        {
            avg_change_rate = linear_slope(infidelity_buffer, buffer_size, buffer_idx);
            avg_change_rate /= infidelity;
        }
        else
        {
            avg_change_rate = 1e3;
        }

        // collect fidelities and change rates from all master process to root process
        if (context->is_master)
        {
            PetscCall(MPI_Gather(&fidelity, 1, MPI_DOUBLE, fidelities, 1, MPI_DOUBLE, 0, context->master_comm));
            PetscCall(MPI_Gather(&avg_change_rate, 1, MPI_DOUBLE, change_rates, 1, MPI_DOUBLE, 0, context->master_comm));
            if (total_rank_id == context->root_id)
            {
                // get max fidelity
                max_fidelity = 0.0;
                max_fidelity_change_rate = 0.0;
                for (int p = 0; p < cnt_parallel; p++)
                {
                    if (fidelities[p] > max_fidelity)
                    {
                        max_fidelity = fidelities[p];
                        max_fidelity_change_rate = change_rates[p];
                    }
                }
                printf("[%5d] Min infidelity: %.6e, change rate: %.6e\n", iter, 1.0 - max_fidelity, max_fidelity_change_rate);
            }
        }

        // broadcast the max fidelity and change rate to all processes
        PetscCall(MPI_Bcast(&max_fidelity, 1, MPI_DOUBLE, context->root_id, PETSC_COMM_WORLD));
        PetscCall(MPI_Bcast(&max_fidelity_change_rate, 1, MPI_DOUBLE, context->root_id, PETSC_COMM_WORLD));

        infidelity = 1.0 - fidelity;

        if ((change_cooldown > change_cooldown_threshold) &&
            (fabs(avg_change_rate) < 1e-4) &&
            (fidelity < max_fidelity))
        {
            PetscPrintf(context->comm, "[%5d] Stream %d: Average change rate too small (%.2e), generating new coupling strength\n",
                        iter, context->stream_id, avg_change_rate);
            is_coupling_reset = 1;
        }
        else
        {
            is_coupling_reset = 0;
        }

        // check if any stream has to reset the coupling strength
        if (context->is_master)
        {
            PetscCall(MPI_Allreduce(&is_coupling_reset, &is_coupling_reset_any, 1, MPI_INT, MPI_LOR, context->master_comm));
            if (is_coupling_reset_any)
            {
                // collect coupling strength from all streams to master
                PetscCall(MPI_Gather(context->coupling_strength, cnt_bond, MPI_DOUBLE, coupling_strength_list, cnt_bond, MPI_DOUBLE, 0, context->master_comm));
                // collect is_coupling_reset from all streams to master
                PetscCall(MPI_Gather(&is_coupling_reset, 1, MPI_INT, is_coupling_reset_list, 1, MPI_INT, 0, context->master_comm));
                if (total_rank_id == context->root_id)
                {
                    sort_index_by_fidelity(fidelities, index_list, cnt_parallel);
                    // set the coupling strength to the average of the best cnt_avg of the contexts
                    int cnt_avg = min_int(cnt_parallel / 4 + 1, 1);
                    for (int i = 0; i < cnt_bond; i++)
                    {
                        double sum = 0.0;
                        for (int j = 0; j < cnt_avg; j++)
                        {
                            sum += coupling_strength_list[index_list[j] * cnt_bond + i];
                        }
                        coupling_strength_reset[i] = sum / cnt_avg;
                    }
                    // send coupling_strength_reset to the processes that need to reset the coupling strength
                    for (int j = 1; j < cnt_parallel; j++)
                    {
                        if (is_coupling_reset_list[j])
                        {
                            PetscCall(MPI_Send(coupling_strength_reset, cnt_bond, MPI_DOUBLE, j, 0, context->master_comm));
                        }
                    }
                }

                if (is_coupling_reset)
                {
                    if (context->master_rank != 0)
                    {
                        PetscCall(MPI_Recv(coupling_strength_reset, cnt_bond, MPI_DOUBLE, 0, 0, context->master_comm, MPI_STATUS_IGNORE));
                    }
                    for (int i = 0; i < cnt_bond; i++)
                    {
                        if (!context->isfixed[i])
                        {
                            coupling_strength_reset[i] = coupling_strength_reset[i] * randn2(context->rng, 1.0, 0.5) + randu2(context->rng, -1.0, 1.0);
                        }
                        else
                        {
                            coupling_strength_reset[i] = context->coupling_strength[i];
                        }
                    }
                }
            }
        }

        // Apply optimizer updates
        if (is_coupling_reset)
        {
            change_cooldown = 0;
            buffer_idx = 0;
            memset(m, 0, cnt_bond * sizeof(double));
            memset(v, 0, cnt_bond * sizeof(double));
            adam_iter = 0;
            PetscCall(set_coupling_strength(context, coupling_strength_reset));
        }
        else
        {
            double lr = learning_rate_schedule(min_int(adam_iter, 2000), 2000, 1e-3, learning_rate);
            PetscCall(adam_optimizer(context, grad, m, v, beta1, beta2, lr, adam_iter + 1));
            adam_iter++;
            change_cooldown++;
        }
    }

    // Cleanup
    free(grad);
    free(m);
    free(v);
    free(infidelity_buffer);
    free(fidelities);
    free(index_list);
    free(coupling_strength_list);
    free(coupling_strength_reset);
    free(is_coupling_reset_list);
    free(change_rates);

    return PETSC_SUCCESS;
}
