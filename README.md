# Quantum Entanglement and Bell Violation Classifier

This repository provides a computational framework to determine the quantum separability boundary and Bell inequality violations in 2-qubit systems using Machine Learning[cite: 34]. It combines a high-performance C++ backend for quantum state generation with a Python-based Machine Learning pipeline for classification and physical analysis.

## Theoretical Background

The physics problem addressed in this project involves characterizing entanglement in bipartite systems described by a $4\times4$ density matrix $\rho$. 
*   **Peres-Horodecki Criterion (PPT)**: A state is separable if and only if its partial transpose $\rho^{T_A}$ has non-negative eigenvalues ($\rho^{T_A} \ge 0$).
*   **Bell's Inequality (CHSH)**: The maximum violation is calculated analytically using the Horodecki criterion, extracting eigenvalues from the correlation matrix to determine if the state violates classical local realism ($S > 2$).
*   **Physical Metrics**: The project calculates State Purity, Von Neumann Entropy, and Entanglement Entropy to analyze the Araki-Lieb inequality and the degree of quantum correlation.

## Repository Structure

*   **`funzioni.h` / `QuantumGenerator`**: C++ class utilizing the `Eigen` library to generate random and separable 4x4 density matrices[cite: 35]. It computes the partial transpose, the PPT criterion, and the Horodecki criterion.
*   **`data_gen.cpp`**: The C++ main application. It generates a balanced dataset of separable and entangled states, extracting 32 raw features (real and imaginary parts of the density matrix) alongside `is_entangled`, `violates_bell`, and `bell_value` labels, saving them to `quantum_data_rich.csv`.
*   **`Makefile`**: Compilation script linking the C++ source files with the Eigen3 library headers (`/usr/include/eigen3`).
*   **Python ML Pipeline (`funzioni.py` / Notebooks)**: Includes functions to reconstruct density matrices from CSV data, calculate entropies, and train models. It features:
    *   Logistic Regression and Support Vector Machines (SVM) with GridSearchCV.
    *   A PyTorch Multilayer Perceptron (`QuantumMLP`) with Early Stopping and Validation.
    *   Tools for Permutation Importance, Confusion Matrices, and Support Vector distribution analysis.

## Prerequisites & Installation

To run this project, you will need:
*   **C++ Compiler**: `g++` with C++11 support.
*   **Eigen3**: C++ template library for linear algebra (expected at `/usr/include/eigen3`).
*   **Python 3.x**:
    *   `numpy`, `pandas`, `matplotlib`, `seaborn`.
    *   `scikit-learn`.
    *   `torch` (PyTorch).

## Usage

**1. Generate the Quantum Dataset**
Compile the C++ code using the provided `Makefile` and run the generator:
```bash
make clean
make all
make run
