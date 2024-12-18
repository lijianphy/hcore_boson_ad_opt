#include <stdio.h>
#include <stdlib.h>
#include <lapacke.h>
#include "hamiltonian.h"
#include "simu_config.h"

static char *generate_output_filename(const char *input_filename)
{
    const char *base_name = strrchr(input_filename, '/');
    base_name = base_name ? base_name + 1 : input_filename;

    const char *last_dot = strrchr(base_name, '.');
    size_t base_len = last_dot ? (size_t)(last_dot - base_name) : strlen(base_name);


    char *output_filename = (char *)malloc(base_len + 32);

    // Copy base name (without extension) and add random string and timestamp
    memcpy(output_filename, base_name, base_len);
    output_filename[base_len] = '\0';

    snprintf(output_filename + base_len, 32, "_eigenvalues.txt");

    return output_filename;
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s <config_file>\n", argv[0]);
        exit(1);
    }

    Simulation_context *context = (Simulation_context *)malloc(sizeof(Simulation_context));
    read_config(argv[1], context);
    print_config(context);

    char *output_filename = generate_output_filename(argv[1]);

    size_t h_dimension = binomial(context->cnt_site, context->cnt_excitation);
    context->h_dimension = h_dimension;
    printf("Hilbert space dimension: %zu\n", h_dimension);

    double *hamiltonian = (double *)malloc(h_dimension * h_dimension * sizeof(double));
    build_hamiltonian_dense(context, hamiltonian);

    // calculate the eivenvalues of the Hamiltonian
    double *eigenvalues = (double *)malloc(h_dimension * sizeof(double));

    // Calculate eigenvalues using LAPACKE
    int info = LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'N', 'U', h_dimension,
                              hamiltonian, h_dimension, eigenvalues);

    if (info > 0)
    {
        fprintf(stderr, "Error: The algorithm failed to compute eigenvalues.\n");
        exit(1);
    }

    FILE *output_file = fopen(output_filename, "w");
    if (output_file == NULL)
    {
        fprintf(stderr, "Error: Unable to open file %s for writing\n", output_filename);
        exit(1);
    }

    for (size_t i = 0; i < h_dimension; i++)
    {
        fprintf(output_file, "%.10lf\n", eigenvalues[i]);
    }
    fclose(output_file);
    free(output_filename);

    free(eigenvalues);
    free(hamiltonian);
    free(context->bonds);
    free(context->coupling_strength);
    free(context);
    return 0;
}
