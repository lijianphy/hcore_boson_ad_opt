#ifndef GRAPH_H
#define GRAPH_H

#include "hamiltonian.h" // for Pair

// Structure for array-based adjacency list
typedef struct
{
    int *neighbors; // Array to store neighbor vertices
    int count;      // Number of neighbors
} AdjVertex;

typedef struct
{
    AdjVertex *vertices; // Array of vertices
    int num_vertices;    // Total number of vertices
} Graph;

// Function declarations
Graph *createGraph(int cnt_site, int cnt_bond, Pair *bonds);
void printGraph(Graph *graph);
void freeGraph(Graph *graph);

#endif // GRAPH_H
