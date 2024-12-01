#include <stdio.h>
#include <stdlib.h>
#include <cJSON.h>
#include "log.h"
#include "hamiltonian.h"

/**
 * @brief Validates the JSON schema for configuration
 * @param json The JSON object to validate
 * @return 1 if the JSON is valid, 0 otherwise
 */
static int validate_json_schema(const cJSON *json)
{
    // Check required fields exist
    const char *required_fields[] = {"cnt_site", "cnt_bond", "cnt_excitation", "bonds", "coupling_strength",
                                     "total_time", "time_steps", "initial_state", "target_state"};
    size_t num_required_fields = sizeof(required_fields) / sizeof(required_fields[0]);
    for (size_t i = 0; i < num_required_fields; i++)
    {
        cJSON *item = cJSON_GetObjectItem(json, required_fields[i]);
        if (item == NULL)
        {
            print_error_msg_mpi("Missing required field: %s", required_fields[i]);
            return 0;
        }
    }

    /// Check types and values
    // Check numbers
    const char *number_fields[] = {"cnt_site", "cnt_bond", "cnt_excitation", "total_time", "time_steps"};
    size_t num_number_fields = sizeof(number_fields) / sizeof(number_fields[0]);
    for (size_t i = 0; i < num_number_fields; i++)
    {
        if (!cJSON_IsNumber(cJSON_GetObjectItem(json, number_fields[i])))
        {
            print_error_msg_mpi("%s must be a number", number_fields[i]);
            return 0;
        }
    }

    // Check number of sites and excitations
    int cnt_site = cJSON_GetObjectItem(json, "cnt_site")->valueint;
    int cnt_bond = cJSON_GetObjectItem(json, "cnt_bond")->valueint;
    int cnt_excitation = cJSON_GetObjectItem(json, "cnt_excitation")->valueint;
    // cnt_site should be less than MAX_SITE
    if (cnt_site >= MAX_SITE)
    {
        print_error_msg_mpi("cnt_site must be less than %d, got %d", MAX_SITE, cnt_site);
        return 0;
    }
    // cnt_excitation should be > 0 and < cnt_site
    if (cnt_excitation <= 0 || cnt_excitation >= cnt_site)
    {
        print_error_msg_mpi("cnt_excitation must be > 0 and < cnt_site, got %d", cnt_excitation);
        return 0;
    }

    // Check arrays
    const char *array_fields[] = {"bonds", "coupling_strength", "initial_state", "target_state"};
    const int array_sizes[] = {cnt_bond, cnt_bond, cnt_excitation, cnt_excitation};
    size_t num_array_fields = sizeof(array_fields) / sizeof(array_fields[0]);
    for (size_t i = 0; i < num_array_fields; i++)
    {
        cJSON *array = cJSON_GetObjectItem(json, array_fields[i]);
        if (!cJSON_IsArray(array))
        {
            print_error_msg_mpi("%s must be an array", array_fields[i]);
            return 0;
        }
        if (cJSON_GetArraySize(array) != array_sizes[i])
        {
            print_error_msg_mpi("%s array size must be %d", array_fields[i], array_sizes[i]);
            return 0;
        }
    }

    // Validate each bond
    cJSON *bonds = cJSON_GetObjectItem(json, "bonds");
    cJSON *strengths = cJSON_GetObjectItem(json, "coupling_strength");
    for (int i = 0; i < cnt_bond; i++)
    {
        cJSON *bond = cJSON_GetArrayItem(bonds, i);
        if (!cJSON_IsArray(bond) || cJSON_GetArraySize(bond) != 2 ||
            !cJSON_IsNumber(cJSON_GetArrayItem(bond, 0)) ||
            !cJSON_IsNumber(cJSON_GetArrayItem(bond, 1)))
        {
            print_error_msg_mpi("Each bond must be an array of 2 integers");
            return 0;
        }
        // first and second elements of bond should be between 0 and cnt_site - 1
        int site1 = cJSON_GetArrayItem(bond, 0)->valueint;
        int site2 = cJSON_GetArrayItem(bond, 1)->valueint;
        if (site1 < 0 || site1 >= cnt_site || site2 < 0 || site2 >= cnt_site)
        {
            print_error_msg_mpi("Elements of bonds must be between 0 and cnt_site - 1");
            return 0;
        }
        if (!cJSON_IsNumber(cJSON_GetArrayItem(strengths, i)))
        {
            print_error_msg_mpi("Each coupling strength must be a number");
            return 0;
        }
    }

    // Elements of bonds should be unique
    for (int i = 0; i < cnt_bond; i++)
    {
        for (int j = i + 1; j < cnt_bond; j++)
        {
            cJSON *bond1 = cJSON_GetArrayItem(bonds, i);
            cJSON *bond2 = cJSON_GetArrayItem(bonds, j);

            int x1 = cJSON_GetArrayItem(bond1, 0)->valueint;
            int y1 = cJSON_GetArrayItem(bond1, 1)->valueint;
            int x2 = cJSON_GetArrayItem(bond2, 0)->valueint;
            int y2 = cJSON_GetArrayItem(bond2, 1)->valueint;

            if ((x1 == x2 && y1 == y2) || (x1 == y2 && y1 == x2))
            {
                print_error_msg_mpi("Elements of bonds must be unique");
                return 0;
            }
        }
    }

    // Validate initial and target states
    cJSON *initial_state = cJSON_GetObjectItem(json, "initial_state");
    cJSON *target_state = cJSON_GetObjectItem(json, "target_state");

    // Validate fixed_couplings and fixed_coupling_strength if they exist
    cJSON *fixed_couplings = cJSON_GetObjectItem(json, "fixed_couplings");
    cJSON *fixed_coupling_strength = cJSON_GetObjectItem(json, "fixed_coupling_strength");
    if (fixed_couplings && fixed_coupling_strength) {
        if (!cJSON_IsArray(fixed_couplings) || !cJSON_IsArray(fixed_coupling_strength)) {
            print_error_msg_mpi("fixed_couplings and fixed_coupling_strength must be arrays");
            return 0;
        }

        int cnt_fixed = cJSON_GetArraySize(fixed_couplings);
        if (cJSON_GetArraySize(fixed_coupling_strength) != cnt_fixed) {
            print_error_msg_mpi("fixed_couplings and fixed_coupling_strength must have the same length");
            return 0;
        }

        // Validate each fixed coupling
        for (int i = 0; i < cnt_fixed; i++) {
            cJSON *coupling = cJSON_GetArrayItem(fixed_couplings, i);
            if (!cJSON_IsArray(coupling) || cJSON_GetArraySize(coupling) != 2) {
                print_error_msg_mpi("Each fixed coupling must be an array of 2 integers");
                return 0;
            }

            int site1 = cJSON_GetArrayItem(coupling, 0)->valueint;
            int site2 = cJSON_GetArrayItem(coupling, 1)->valueint;
            
            // Check if this fixed bond exists in the bonds array
            int found = 0;
            for (int j = 0; j < cnt_bond; j++) {
                cJSON *bond = cJSON_GetArrayItem(bonds, j);
                int bsite1 = cJSON_GetArrayItem(bond, 0)->valueint;
                int bsite2 = cJSON_GetArrayItem(bond, 1)->valueint;
                if ((site1 == bsite1 && site2 == bsite2) || (site1 == bsite2 && site2 == bsite1)) {
                    found = 1;
                    break;
                }
            }
            if (!found) {
                print_error_msg_mpi("Fixed coupling [%d, %d] not found in bonds", site1, site2);
                return 0;
            }

            if (!cJSON_IsNumber(cJSON_GetArrayItem(fixed_coupling_strength, i))) {
                print_error_msg_mpi("Each fixed coupling strength must be a number");
                return 0;
            }
        }
    }

    // array elements should be integers and between 0 and cnt_site - 1
    for (int i = 0; i < cnt_excitation; i++)
    {
        if (!cJSON_IsNumber(cJSON_GetArrayItem(initial_state, i)) ||
            !cJSON_IsNumber(cJSON_GetArrayItem(target_state, i)))
        {
            print_error_msg_mpi("Initial and target states must be arrays of integers");
            return 0;
        }
        int initial_site = cJSON_GetArrayItem(initial_state, i)->valueint;
        int target_site = cJSON_GetArrayItem(target_state, i)->valueint;
        if (initial_site < 0 || initial_site >= cnt_site || target_site < 0 || target_site >= cnt_site)
        {
            print_error_msg_mpi("Initial and target states must be between 0 and cnt_site - 1");
            return 0;
        }
    }

    // elements of initial state and target state should be unique
    for (int i = 0; i < cnt_excitation; i++)
    {
        for (int j = i + 1; j < cnt_excitation; j++)
        {
            if (cJSON_GetArrayItem(initial_state, i)->valueint == cJSON_GetArrayItem(initial_state, j)->valueint ||
                cJSON_GetArrayItem(target_state, i)->valueint == cJSON_GetArrayItem(target_state, j)->valueint)
            {
                print_error_msg_mpi("Elements of initial and target states must be unique");
                return 0;
            }
        }
    }

    return 1;
}

/**
 * @brief Reads configuration from a JSON file and sets the configuration in the context
 * @param file_name Path to the configuration JSON file
 * @param context Pointer to the simulation context to be populated
 * @return 0 on success, 1 on failure
 */
int read_config(const char *file_name, Simulation_context *context)
{
    // Read the entire file into a string
    FILE *file = fopen(file_name, "r");
    if (file == NULL)
    {
        print_error_msg_mpi("Unable to open file %s", file_name);
        return 1;
    }

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    rewind(file);

    char *json_str = (char *)malloc(file_size + 1);
    if (json_str == NULL)
    {
        print_error_msg_mpi("Unable to allocate memory");
        fclose(file);
        return 1;
    }
    long s = fread(json_str, 1, file_size, file);
    if (s != file_size)
    {
        print_error_msg_mpi("Error reading file %s", file_name);
        free(json_str);
        fclose(file);
        return 1;
    }
    json_str[file_size] = '\0';
    fclose(file);

    // Parse JSON
    cJSON *json = cJSON_Parse(json_str);
    if (json == NULL)
    {
        const char *error_ptr = cJSON_GetErrorPtr();
        if (error_ptr != NULL)
        {
            print_error_msg_mpi("Error parsing JSON: %s", error_ptr);
        }
        free(json_str);
        return 1;
    }

    // Validate JSON schema
    if (!validate_json_schema(json))
    {
        cJSON_Delete(json);
        free(json_str);
        return 1;
    }

    // Extract configuration
    context->cnt_site = cJSON_GetObjectItem(json, "cnt_site")->valueint;
    context->cnt_bond = cJSON_GetObjectItem(json, "cnt_bond")->valueint;
    context->cnt_excitation = cJSON_GetObjectItem(json, "cnt_excitation")->valueint;

    // Read bonds and coupling_strength
    context->bonds = (Pair *)malloc(context->cnt_bond * sizeof(Pair));
    context->coupling_strength = (double *)malloc(context->cnt_bond * sizeof(double));

    cJSON *bonds = cJSON_GetObjectItem(json, "bonds");
    cJSON *strengths = cJSON_GetObjectItem(json, "coupling_strength");
    for (int i = 0; i < context->cnt_bond; i++)
    {
        cJSON *bond = cJSON_GetArrayItem(bonds, i);
        context->bonds[i].x = cJSON_GetArrayItem(bond, 0)->valueint;
        context->bonds[i].y = cJSON_GetArrayItem(bond, 1)->valueint;
        context->coupling_strength[i] = cJSON_GetArrayItem(strengths, i)->valuedouble;
    }

    // Read total_time, time_steps, initial_state, target_state
    context->total_time = cJSON_GetObjectItem(json, "total_time")->valuedouble;
    context->time_steps = cJSON_GetObjectItem(json, "time_steps")->valueint;
    context->initial_state = 0;
    context->target_state = 0;
    cJSON *initial_state = cJSON_GetObjectItem(json, "initial_state");
    cJSON *target_state = cJSON_GetObjectItem(json, "target_state");
    for (int i = 0; i < context->cnt_excitation; i++)
    {
        context->initial_state |= (State)1 << cJSON_GetArrayItem(initial_state, i)->valueint;
        context->target_state |= (State)1 << cJSON_GetArrayItem(target_state, i)->valueint;
    }

    // Read fixed coupling configuration if it exists
    context->isfixed = (int *)calloc(context->cnt_bond, sizeof(int));
    cJSON *fixed_couplings = cJSON_GetObjectItem(json, "fixed_couplings");
    cJSON *fixed_coupling_strength = cJSON_GetObjectItem(json, "fixed_coupling_strength");
    
    if (fixed_couplings && fixed_coupling_strength) {
        int cnt_fixed = cJSON_GetArraySize(fixed_couplings);
        
        // For each fixed coupling, find its index in the bonds array
        for (int i = 0; i < cnt_fixed; i++) {
            cJSON *coupling = cJSON_GetArrayItem(fixed_couplings, i);
            int site1 = cJSON_GetArrayItem(coupling, 0)->valueint;
            int site2 = cJSON_GetArrayItem(coupling, 1)->valueint;
            double fixed_strength = cJSON_GetArrayItem(fixed_coupling_strength, i)->valuedouble;
            
            // Find the corresponding bond index
            for (int j = 0; j < context->cnt_bond; j++) {
                if ((context->bonds[j].x == site1 && context->bonds[j].y == site2) ||
                    (context->bonds[j].x == site2 && context->bonds[j].y == site1)) {
                    context->isfixed[j] = 1;
                    context->coupling_strength[j] = fixed_strength;
                    break;
                }
            }
        }
    }

    // Cleanup
    cJSON_Delete(json);
    free(json_str);
    return 0;
}

/**
 * @brief Prints the current configuration settings
 * @param context Pointer to the simulation context containing the configuration
 * @return Always returns 0
 */
int print_config(const Simulation_context *context)
{
    printf("cnt_site: %d\n", context->cnt_site);
    printf("cnt_bond: %d\n", context->cnt_bond);
    printf("cnt_excitation: %d\n", context->cnt_excitation);

    for (int i = 0; i < context->cnt_bond; i++)
    {
        printf("bond[%d]: %d %d %lf\n", i, context->bonds[i].x, context->bonds[i].y, context->coupling_strength[i]);
    }

    printf("total_time: %lf\n", context->total_time);
    printf("time_steps: %d\n", context->time_steps);
    printf("initial_state: ");
    print_state(context->initial_state, context->cnt_site);
    printf("target_state:  ");
    print_state(context->target_state, context->cnt_site);

    return 0;
}
