#include <stdio.h>
#include <stdlib.h>
#include "graph.h"

// Function to create graph
Graph* createGraph(int cnt_site, int cnt_bond, Pair *bonds) {
    if (cnt_site <= 0 || cnt_bond <= 0 || bonds == NULL) {
        return NULL;
    }

    // Allocate graph structure
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    if (!graph) return NULL;
    
    graph->num_vertices = cnt_site;
    graph->vertices = (AdjVertex*)calloc(cnt_site, sizeof(AdjVertex));
    if (!graph->vertices) {
        free(graph);
        return NULL;
    }

    // First pass: count neighbors for each vertex
    for (int i = 0; i < cnt_bond; i++) {
        graph->vertices[bonds[i].x].count++;
        graph->vertices[bonds[i].y].count++;
    }

    // Allocate arrays for neighbors
    for (int i = 0; i < cnt_site; i++) {
        graph->vertices[i].neighbors = (int*)malloc(graph->vertices[i].count * sizeof(int));
        if (!graph->vertices[i].neighbors) {
            // Cleanup on failure
            for (int j = 0; j < i; j++) {
                free(graph->vertices[j].neighbors);
            }
            free(graph->vertices);
            free(graph);
            return NULL;
        }
        graph->vertices[i].count = 0; // Reset count for second pass
    }

    // Second pass: fill neighbor arrays
    for (int i = 0; i < cnt_bond; i++) {
        int src = bonds[i].x;
        int dst = bonds[i].y;
        
        graph->vertices[src].neighbors[graph->vertices[src].count++] = dst;
        graph->vertices[dst].neighbors[graph->vertices[dst].count++] = src;
    }

    return graph;
}

// Function to print graph
void printGraph(Graph* graph) {
    if (!graph) return;
    
    for (int i = 0; i < graph->num_vertices; i++) {
        printf("Vertex %d:", i);
        for (int j = 0; j < graph->vertices[i].count; j++) {
            printf(" %d", graph->vertices[i].neighbors[j]);
        }
        printf("\n");
    }
}

// Function to free graph
void freeGraph(Graph* graph) {
    if (!graph) return;
    
    for (int i = 0; i < graph->num_vertices; i++) {
        free(graph->vertices[i].neighbors);
    }
    free(graph->vertices);
    free(graph);
}
