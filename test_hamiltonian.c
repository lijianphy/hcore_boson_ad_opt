/**
 * @file test_hamiltonian.c
 * @brief Test the eigenvalue calculations of a Hamiltonian matrix, comparing dense and sparse methods
 */

#include "hamiltonian.h"
#include <stdlib.h>   // malloc, free
#include <lapacke.h>  // LAPACKE_dsyev
#include <petscmat.h> // Mat
#include <slepceps.h> // EPS
#include "log.h"      // print_error_msg_mpi, printf_master

/**
 * @brief Calculates eigenvalues of the dense Hamiltonian matrix
 * @param context Pointer to the simulation context
 * @param hamiltonian Pointer to the dense matrix (will be modified)
 * @return The largest eigenvalue
 */
double eigen_values_dense(const Simulation_context *context, double *hamiltonian)
{
    size_t h_dimension = context->h_dimension;
    double *eigen_values = (double *)malloc(h_dimension * sizeof(double));
    int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'N', 'U', h_dimension, hamiltonian, h_dimension, eigen_values);
    if (info != 0)
    {
        print_error_msg_mpi("Failed to compute eigenvalues");
        exit(1);
    }

    printf_master("Eigenvalues:\n");
    for (size_t i = 0; i < h_dimension; i++)
    {
        printf_master("%lf\n", eigen_values[i]);
    }

    double largest = eigen_values[h_dimension - 1];
    free(eigen_values);
    return largest;
}

/**
 * @brief Calculates the largest eigenvalue of a sparse matrix using SLEPc
 * @param A The sparse matrix
 * @param eigenvalue Pointer to store the largest eigenvalue
 * @return PetscErrorCode
 */
PetscErrorCode largest_eigenvalue_sparse(Mat A, PetscReal *eigenvalue)
{
    EPS eps;
    EPSType type;
    PetscInt nev = 1; // We only want the largest eigenvalue
    PetscInt max_it;
    PetscReal error;

    // Create the eigenvalue solver context
    PetscCall(EPSCreate(PETSC_COMM_WORLD, &eps));

    // Set operators. For a standard eigenvalue problem, only A is needed
    PetscCall(EPSSetOperators(eps, A, NULL));
    PetscCall(EPSSetProblemType(eps, EPS_HEP)); // Hermitian eigenvalue problem

    // Set solver parameters
    PetscCall(EPSSetWhichEigenpairs(eps, EPS_LARGEST_REAL));
    PetscCall(EPSSetDimensions(eps, nev, PETSC_DEFAULT, PETSC_DEFAULT));

    // Set solver tolerances
    PetscCall(EPSSetTolerances(eps, 1e-8, 100));

    // Set up the solver
    PetscCall(EPSSetFromOptions(eps));

    // Solve the eigenvalue problem
    PetscCall(EPSSolve(eps));

    // Get number of converged eigenvalues
    PetscInt nconv;
    PetscCall(EPSGetConverged(eps, &nconv));
    if (nconv < 1)
    {
        PetscPrintf(PETSC_COMM_WORLD, "No eigenvalues converged!\n");
        return 1;
    }

    // Get the largest eigenvalue
    PetscScalar eigr; // Real part
    PetscScalar eigi; // Imaginary part (should be zero for symmetric matrix)
    PetscCall(EPSGetEigenpair(eps, 0, &eigr, &eigi, NULL, NULL));
    *eigenvalue = PetscRealPart(eigr);

    // Get solver information
    PetscCall(EPSGetIterationNumber(eps, &max_it));
    PetscCall(EPSGetType(eps, &type));
    PetscCall(EPSGetErrorEstimate(eps, 0, &error));

    // Print solver information
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Solution method: %s\n", type));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Number of iterations: %d\n", max_it));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Error estimate: %g\n", (double)error));

    // Clean up
    PetscCall(EPSDestroy(&eps));

    return PETSC_SUCCESS;
}

/**
 * @brief Tests eigenvalue calculations by comparing dense and sparse matrix results
 * @param context Pointer to the simulation context
 * @return PetscErrorCode
 *
 * Compares eigenvalues computed using dense and sparse matrix methods and
 * prints the results and relative difference.
 */
PetscErrorCode test_eigenvalues(const Simulation_context *context)
{
    printf_master("Building dense Hamiltonian matrix\n");
    // allocate memory for dense Hamiltonian
    size_t h_dimension = context->h_dimension;
    double *h_dense = (double *)malloc(h_dimension * h_dimension * sizeof(double));
    if (h_dense == NULL)
    {
        print_error_msg_mpi("Unable to allocate memory for dense Hamiltonian");
        exit(1);
    }
    build_hamiltonian_dense(context, h_dense);
    printf_master("Computing dense eigenvalues\n");
    double largest_dense = eigen_values_dense(context, h_dense);

    printf_master("Computing largest sparse eigenvalue\n");
    PetscReal largest_sparse;
    PetscCall(largest_eigenvalue_sparse(context->hamiltonian, &largest_sparse));

    // Compare results
    printf_master("\n=== Results Comparison ===\n");
    printf_master("Largest eigenvalue (dense):  %lf\n", largest_dense);
    printf_master("Largest eigenvalue (sparse): %lf\n", (double)largest_sparse);
    printf_master("Relative difference: %e\n", fabs((largest_dense - largest_sparse) / largest_dense));

    free(h_dense);
    return PETSC_SUCCESS;
}

int main(int argc, char *argv[])
{
    const char *help = "Solve the largest eigenvalue of a Hamiltonian matrix\n";
    // Initialize PETSc and SLEPc
    PetscCall(SlepcInitialize(&argc, &argv, NULL, help));

    if (argc != 2)
    {
        print_error_msg_mpi("Usage: %s <config_file>", argv[0]);
        exit(1);
    }

    Simulation_context context = {0}; // Initialize all fields to 0
    PetscCall(init_simulation_context(&context, argv[1]));
    PetscCall(test_eigenvalues(&context));

    // Clean up
    printf_master("\nCleaning up\n");
    PetscCall(free_simulation_context(&context));
    PetscCall(SlepcFinalize());
    return 0;
}
