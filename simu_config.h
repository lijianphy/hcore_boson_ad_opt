#ifndef SIMU_CONFIG_H_
#define SIMU_CONFIG_H_

#include "hamiltonian.h"

/**
 * @brief Reads configuration from a JSON file and sets the configuration in the context
 * @param file_name Path to the configuration JSON file
 * @param context Pointer to the simulation context to be populated
 * @throw Exits with error code 1 if file operations or JSON parsing fails
 */
int read_config(const char *file_name, Simulation_context *context);

/**
 * @brief Prints the current configuration settings
 * @param context Pointer to the simulation context containing the configuration
 * @return Always returns 0
 */
int print_config(const Simulation_context *context);

#endif // SIMU_CONFIG_H_
