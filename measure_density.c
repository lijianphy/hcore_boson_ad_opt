#include <petsc.h>
#include <slepc.h>
#include "hamiltonian.h"
#include "evolution_ad.h"
#include "log.h"

PetscErrorCode init_density_operator(Simulation_context *context, Vec *density_operator_list) {
    int cnt_site = context->cnt_site;
    int cnt_excitation = context->cnt_excitation;

    for (int i = 0; i < cnt_site; i++) {
        PetscCall(VecDuplicate(context->init_vec, &density_operator_list[i]));
        PetscCall(VecSet(density_operator_list[i], 0.0));
    }

    size_t mpi_local_begin = context->local_partition_begin;
    size_t mpi_local_end = context->local_partition_begin + context->local_partition_size;
    State current_state = index2state(cnt_site, cnt_excitation, mpi_local_begin);
    for (size_t i = mpi_local_begin; i < mpi_local_end; i++, current_state = next_state(current_state)) {
        for (int pos = 0; pos < cnt_site; pos++) {
            if (is_bit_set(current_state, pos)) {
                PetscCall(VecSetValue(density_operator_list[pos], i, 1.0, INSERT_VALUES));
            }
        }
    }
    
    for (int pos = 0; pos < cnt_site; pos++) {
        PetscCall(VecAssemblyBegin(density_operator_list[pos]));
        PetscCall(VecAssemblyEnd(density_operator_list[pos]));
    }

    return PETSC_SUCCESS;
}

static void output_density(Simulation_context *context, int time_step, double *densities) {
    FILE *fp = context->output_file;
    if (fp && context->partition_id == 0) {
        fprintf(fp, "{\"time_step\": %d, \"density\": [", time_step);
        for (int i = 0; i < context->cnt_site; i++) {
            fprintf(fp, "%.10e%s", densities[i], i < context->cnt_site - 1 ? ", " : "");
        }
        fprintf(fp, "]}\n");
        fflush(fp);
    }
}

PetscErrorCode measure_density(Simulation_context *context, Vec *density_operator_list) {
    Vec tmp;
    PetscScalar result;
    double *densities = (double *)malloc(context->cnt_site * sizeof(double));
    PetscCall(VecDuplicate(context->init_vec, &tmp));
    PetscCall(forward_evolution(context));
    for (int i = 0; i <= context->time_steps; i++) {
        for (int pos = 0; pos < context->cnt_site; pos++) {
            Vec f = context->forward_path[i];
            Vec d = density_operator_list[pos];
            PetscCall(VecPointwiseMult(tmp, f, d));
            PetscCall(VecDot(tmp, f, &result));
            densities[pos] = PetscRealPart(result);
        }
        output_density(context, i, densities);
    }
    PetscCall(VecDestroy(&tmp));
    free(densities);
    return PETSC_SUCCESS;
}

int main(int argc, char **argv) {
    const char *help = "Measure the density during a single evolution\n";
    // Initialize PETSc and SLEPc
    PetscCall(SlepcInitialize(&argc, &argv, NULL, help));
    PetscCheckAbort(argc == 2, PETSC_COMM_WORLD, PETSC_ERR_FILE_OPEN, "Usage: %s <config_file>", argv[0]);

    Simulation_context *context = (Simulation_context *)malloc(sizeof(Simulation_context));
    PetscCall(init_simulation_context(context, argv[1]));

    // allocate memory for the density operator list
    Vec *density_operator_list = (Vec *)malloc(context->cnt_site * sizeof(Vec));
    PetscCall(init_density_operator(context, density_operator_list));
    PetscCall(measure_density(context, density_operator_list));

    // Clean up
    printf_master("\nCleaning up\n");
    for (int i = 0; i < context->cnt_site; i++) {
        PetscCall(VecDestroy(&density_operator_list[i]));
    }
    free(density_operator_list);
    PetscCall(free_simulation_context(context));
    free(context);
    PetscCall(SlepcFinalize());
    return 0;
}