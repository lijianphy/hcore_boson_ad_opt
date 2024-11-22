#include <stdlib.h>
#include <string.h>
#include "graph_csr.h"

/**
 * Create a CSR matrix from a list of bonds and coupling strengths.
 */
csr_matrix* bonds_to_csr(int num_vertices, int num_bonds, Pair *bonds, double* coupling_strength) {
    csr_matrix* csr = malloc(sizeof(csr_matrix));
    if (!csr) return NULL;

    csr->num_vertices = num_vertices;
    csr->num_nonzero = num_bonds * 2;  // Undirected graph, each bond creates 2 entries
    
    // Allocate memory
    csr->row_ptr = calloc(num_vertices + 1, sizeof(int));
    csr->col_ind = malloc(csr->num_nonzero * sizeof(int));
    csr->values = malloc(csr->num_nonzero * sizeof(double));
    
    if (!csr->row_ptr || !csr->col_ind || !csr->values) {
        free(csr->row_ptr);
        free(csr->col_ind);
        free(csr->values);
        free(csr);
        return NULL;
    }

    // Count number of nonzeros per row
    int* row_counts = calloc(num_vertices, sizeof(int));
    for (int i = 0; i < num_bonds; i++) {
        row_counts[bonds[i].x]++;
        row_counts[bonds[i].y]++;
    }

    // Calculate row_ptr
    csr->row_ptr[0] = 0;
    for (int i = 0; i < num_vertices; i++) {
        csr->row_ptr[i + 1] = csr->row_ptr[i] + row_counts[i];
    }

    // Reset counters for filling col_ind and values
    memset(row_counts, 0, num_vertices * sizeof(int));

    // Fill col_ind and values
    for (int i = 0; i < num_bonds; i++) {
        int row1 = bonds[i].x;
        int row2 = bonds[i].y;
        double val = coupling_strength[i];

        // Add edge row1 -> row2
        int idx1 = csr->row_ptr[row1] + row_counts[row1];
        csr->col_ind[idx1] = row2;
        csr->values[idx1] = val;
        row_counts[row1]++;

        // Add edge row2 -> row1
        int idx2 = csr->row_ptr[row2] + row_counts[row2];
        csr->col_ind[idx2] = row1;
        csr->values[idx2] = val;
        row_counts[row2]++;
    }

    free(row_counts);
    return csr;
}

// Helper function to free CSR matrix
void free_csr(csr_matrix* csr) {
    if (csr) {
        free(csr->row_ptr);
        free(csr->col_ind);
        free(csr->values);
        free(csr);
    }
}
