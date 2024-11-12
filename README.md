This code performs quantum optimal control using automatic differentiation (AD) to find the optimal coupling strengths for driving a system of hardcore bosons from an initial state to a target state. It leverages PETSc and SLEPc for efficient sparse matrix operations and eigenvalue calculations, and uses CUnit for unit testing.

## Features

* **Hamiltonian Construction:** Constructs the Hamiltonian matrix for a system of hardcore bosons, supporting both dense and sparse representations. The system's connectivity (geometry of coupling) is defined via a JSON configuration file.
* **Time Evolution:** Simulates the real-time evolution of the quantum system using the matrix exponential, calculated with Expokit through SLEPc's `MFN` interface.
* **Automatic Differentiation:** Implements AD to compute gradients of a target function (the distance between the final evolved state and the desired target state) with respect to the coupling strengths.
* **Optimization:** Optimizes the coupling strengths using gradient descent and the Adam optimizer.

## Dependencies

* **PETSc:** For sparse matrix operations and parallel computing.
* **SLEPc:** For eigenvalue problems and matrix functions (specifically Expokit for matrix exponential).
* **CBLAS:** For basic linear algebra subprograms.
* **LAPACKE:** For linear algebra routines.
* **libcJSON:** For parsing JSON configuration files.
* **CUnit:** For unit testing.
* **MPI:** For parallel communication (mpich).
* **OpenMP:** For shared-memory parallelism within nodes.

## Build Instructions

The code is built using a Makefile. On macOS, you'll need to adjust the `PETSC_DIR`, `PETSC_ARCH`, and `SLEPC_DIR` variables in the Makefile to match your PETSc and SLEPc installation paths.  On Linux, ensure you have the required packages installed.  Then, navigate to the project directory and run:

```bash
make
```

This will compile all source files and create the executables.

## Running Tests

To run the unit tests, execute:

```bash
make test
```

This will compile and run the unit test executables and report the results.

## Running the Optimization

To run the optimization, execute:

```bash
mpirun -np <number_of_processes> ./test_optimize <config_file>
```

Replace `<number_of_processes>` with the desired number of MPI processes and `<config_file>` with the path to the JSON configuration file.  A few of sample configuration files can be found in the `config` directory.

## Configuration File Format

The configuration file is a JSON file that specifies the system parameters, including the number of sites, bonds, excitations, the bond connections, initial coupling strengths, total evolution time, time steps, initial state, and target state.  See `config/1d_8_4_v1.json` for an example.  Key parameters include:

* `"cnt_site"`: Number of sites in the system.
* `"cnt_bond"`: Number of bonds between sites.
* `"cnt_excitation"`: Number of excitations (hardcore bosons) in the system.
* `"bonds"`: An array of bond pairs, where each pair is an array of two site indices (0-indexed).
* `"coupling_strength"`: An array of initial coupling strengths for each bond.
* `"total_time"`: Total time for the evolution.
* `"time_steps"`: Number of time steps for the evolution.
* `"initial_state"`: An array of site indices representing the initially occupied sites (a fock state).
* `"target_state"`: An array of site indices representing the desired final occupied sites (another fock state).


## Code Structure

* **`bits.h`, `bits.c`**: Bit manipulation functions.
* **`combination.h`, `combination.c`**: Combinatorial calculations (binomial coefficients, permutations).
* **`log.h`**: Logging macros.
* **`math_constant.h`**: Mathematical constants.
* **`vec_math.h`, `vec_math.c`**: Vectorized math operations.
* **`hamiltonian.h`, `hamiltonian.c`**: Functions for building and managing the Hamiltonian matrix.
* **`evolution_ad.h`, `evolution_ad.c`**: Core functions for time evolution, gradient calculation, and optimization.
* **`test_*.c`**: Unit test files.
* **`test_ad.c`**: Example of calculating gradients using AD.
* **`test_optimize.c`**: Example of optimizing coupling strengths.
* **`test_hamiltonian.c`**: Example of calculating eigenvalues.
* **`Makefile`**: Build system.

