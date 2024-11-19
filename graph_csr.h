#ifndef GRAPH_CSR_H
#define GRAPH_CSR_H

#include "hamiltonian.h" // for Pair struct

// CSR matrix representation
typedef struct
{
    int *row_ptr;   // Size: num_vertices + 1
    int *col_ind;   // Size: num_nonzero
    double *values; // Size: num_nonzero
    int num_vertices;
    int num_nonzero;
} csr_matrix;

// Convert bonds to CSR format
csr_matrix *bonds_to_csr(int num_vertices, int num_bonds, Pair *bonds, double *coupling_strength);

// Free CSR matrix
void free_csr(csr_matrix *csr);

#endif // GRAPH_CSR_H
