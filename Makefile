# =============================================================================
# Configuration Section
# =============================================================================
# PETSc and SLEPc are scientific computing libraries
PLATFORM := $(shell uname)
ifeq ($(PLATFORM), Darwin)
  # macOS
  PETSC_DIR := /Users/lijian/projects/petsc-3.22.1
  PETSC_ARCH := macos-gnu-complex
  SLEPC_DIR := /Users/lijian/projects/slepc-3.22.1
  petsc.pc := $(PETSC_DIR)/$(PETSC_ARCH)/lib/pkgconfig/PETSc.pc
  slepc.pc := $(SLEPC_DIR)/$(PETSC_ARCH)/lib/pkgconfig/SLEPc.pc
  PACKAGES := $(petsc.pc) $(slepc.pc) openblas libcjson cunit mpich
else
  # Linux
  PACKAGES := petsc slepc blas lapacke libcjson cunit mpi
endif

# =============================================================================
# Compiler Settings
# =============================================================================
# Use MPI C compiler wrapper
CC := mpicc
# Compiler flags:
#   -Wall, -Wextra: Enable comprehensive warning messages
#   -std=c17: Use C17 standard
#   -pedantic: Strict ISO C conformance
#   -O2: Optimization level 2
#   -fopenmp: Enable OpenMP support
CFLAGS := -Wall -Wextra -std=c17 -pedantic -O2 -fopenmp $(shell pkg-config --cflags $(PACKAGES))
# Linker flags - include all required libraries
LDFLAGS := $(shell pkg-config --libs $(PACKAGES)) -lm

# =============================================================================
# Source Files and Targets
# =============================================================================
# List all source files
SRCS := $(wildcard *.c)
# Generate object file names by replacing .c with .o
OBJS := $(SRCS:.c=.o)
# List of test executables
TEST_TARGETS := test_ad test_bits test_cblas test_combination test_hamiltonian test_vec_math test_optimize
# All final executables
TARGETS := $(TEST_TARGETS)

# =============================================================================
# Primary Targets (Phony Targets)
# =============================================================================
# Phony targets are targets that don't create files with the same name
.PHONY: all clean test print

# Default target: build all executables
all: $(TARGETS)

# Run all tests
test: $(TEST_TARGETS)
#	./test_bits
#	./test_combination
#	./test_vec_math
#	mpirun -np 4 ./test_hamiltonian config/8_4_v1.json
#	mpirun -np 4 ./test_ad config/8_4_v1.json
	mpirun -np 4 ./test_optimize config/8_4_v1.json

# Print important variables for debugging Makefile
print:
	@echo "CFLAGS = $(CFLAGS)"
	@echo "LDFLAGS = $(LDFLAGS)"
	@echo "TARGETS = $(TARGETS)"
	@echo "SRCS = $(SRCS)"

# =============================================================================
# Build Rules
# =============================================================================
# Pattern rule for object files
# $<: first prerequisite
%.o: %.c bits.h log.h
	$(CC) $(CFLAGS) -c $< -o $@

# Individual target rules
# $^: all prerequisites
# $@: target name
# $(LDFLAGS) at the end links all required libraries
test_bits: test_bits.o
	$(CC) $^ -o $@ $(LDFLAGS)

test_combination: test_combination.o combination.o
	$(CC) $^ -o $@ $(LDFLAGS)

test_vec_math: test_vec_math.o vec_math.o
	$(CC) $^ -o $@ $(LDFLAGS)

test_hamiltonian: test_hamiltonian.o hamiltonian.o combination.o
	$(CC) $^ -o $@ $(LDFLAGS)

test_ad: test_ad.o evolution_ad.o hamiltonian.o combination.o
	$(CC) $^ -o $@ $(LDFLAGS)

test_cblas: test_cblas.o
	$(CC) $^ -o $@ $(LDFLAGS)

test_optimize: test_optimize.o evolution_ad.o hamiltonian.o combination.o vec_math.o
	$(CC) $^ -o $@ $(LDFLAGS)

# Clean all generated files
clean:
	rm -f $(OBJS) $(TARGETS)
