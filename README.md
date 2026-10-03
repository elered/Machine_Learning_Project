# Quantum Entanglement Classification using SVMs and Neural Networks

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-red.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)](https://scikit-learn.org/)
[![Eigen](https://img.shields.io/badge/C++-Eigen_Library-lightgrey.svg)](https://eigen.tuxfamily.org/)

This repository contains the implementation and source code for the academic project **"Quantum Entanglement Classification via SVMs and Neural Networks"**, developed at the *Università degli Studi di Milano*. The project explores machine learning and deep learning approaches to distinguish between separable and entangled quantum states in bipartite 2-qubit systems.

---

## Repository Structure

- `funzioni.h`: C++ core library implementing quantum state generation (density matrices), partial transposition, the PPT criterion, and CHSH inequality/Bell value calculations using the *Eigen* library.
- `data_gen.cpp`: C++ script to generate the balanced dataset of quantum states.
- `Makefile`: Compilation configuration for the C++ generator.
- `Project.ipynb`: Main Jupyter Notebook covering exploratory data analysis, PCA, hyperparameter tuning via Optuna, SVM (RBF kernel) classification, and Multilayer Perceptron (MLP) training.
- `project_presentation`: Pdf presentation for the project.

---

## Theoretical Background & Dataset Generation

The classification task focuses on bipartite 2-qubit density operators $\rho$ ($4 \times 4$ statistical matrices) satisfying positivity ($\rho \ge 0$) and unit trace ($\text{Tr}[\rho] = 1$):

1. **Separable States:** Generated as a tensor product of two independent $2 \times 2$ subsystem density matrices ($\rho_{\text{sep}} = \rho_A \otimes \rho_B$).
2. **Entangled States:** Generated randomly in the full $4 \times 4$ Hilbert space and rigorously filtered using the **Peres-Horodecki PPT (Positive Partial Transpose)** criterion ($\rho^{T_A} \ge 0$) to ensure a balanced dataset of true entangled instances.

Each sample is described by **32 raw features** (real and imaginary components of the density matrix) alongside physical validation metrics like the **CHSH Bell inequality violation value ($S$)**.

---

## Methodology & Pipeline

1. **Dimensionality Reduction (PCA):** 
   - Analyzed feature multicollinearity stemming from physical constraints ($\text{Tr}[\rho]=1$, Hermiticity).
   - Applied **Principal Component Analysis (PCA)**, retaining **14 principal components** to capture **>95% of the cumulative variance**.
2. **Support Vector Machine (SVM):**
   - Utilized an RBF kernel ($K(x, x') = e^{-\gamma \vert{}\vert{}x - x'\vert{}\vert{}^2}$).
   - Hyperparameters ($C$ and $\gamma$) optimized via **Optuna** using 5-fold cross-validation.
3. **Multilayer Perceptron (MLP):**
   - Designed a feed-forward neural network taking the 14 PCA components as input, featuring a hidden layer with ReLU activation and a Softmax output layer.
   - Optimized architecture size, learning rate, and weight decay using Optuna.

---

## Getting Started

### Prerequisites
Make sure you have a C++ compiler with the **Eigen3** library installed, alongside a Python environment with the required packages.

```bash
# Clone the repository
git clone [https://github.com/your-username/quantum-entanglement-ml.gitt](https://github.com/your-username/quantum-entanglement-ml.gitt)
cd quantum-entanglement-ml

# Python dependencies
pip install pandas numpy matplotlib seaborn scikit-learn torch optuna
