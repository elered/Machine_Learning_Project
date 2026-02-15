#ifndef QUANTUM_GENERATOR_H
#define QUANTUM_GENERATOR_H

#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include <complex>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace Eigen;

typedef std::complex<double> Complex;

class QuantumGenerator {
private:
    Matrix2cd I, sx, sy, sz;
    Matrix2cd pauli[4]; // 0:I, 1:X, 2:Y, 3:Z

public:
    QuantumGenerator() {
        I = Matrix2cd::Identity();
        sx << 0, 1, 1, 0;
        sy << 0, Complex(0, -1), Complex(0, 1), 0;
        sz << 1, 0, 0, -1;
        
        pauli[0] = I; 
        pauli[1] = sx; 
        pauli[2] = sy; 
        pauli[3] = sz;
    }

    // Genera una matrice densità 4x4 casuale (potenzialmente entangled)
    Matrix4cd generateRandomDM() {
        Matrix4cd G = Matrix4cd::Random();
        Matrix4cd rho = G * G.adjoint(); // Assicura che sia Hermitiana e definita positiva
        return rho / rho.trace();        // Normalizzazione traccia = 1
    }

    // Genera una matrice densità 4x4 separabile (prodotto di due stati 2x2)
    Matrix4cd generateSeparableDM() {
        Matrix2cd Ga = Matrix2cd::Random();
        Matrix2cd Gb = Matrix2cd::Random();
        Matrix2cd rhoA = Ga * Ga.adjoint(); 
        rhoA /= rhoA.trace();
        Matrix2cd rhoB = Gb * Gb.adjoint(); 
        rhoB /= rhoB.trace();
        return kroneckerProduct(rhoA, rhoB);
    }

    // Calcola la Trasposta Parziale rispetto al secondo qubit
    Matrix4cd partialTranspose(const Matrix4cd& rho) {
        Matrix4cd res = Matrix4cd::Zero();
        for (int i=0; i<2; ++i) {
            for (int j=0; j<2; ++j) {
                for (int k=0; k<2; ++k) {
                    for (int l=0; l<2; ++l) {
                        res(2*i + l, 2*k + j) = rho(2*i + j, 2*k + l);
                    }
                }
            }
        }
        return res;
    }

    // Criterio PPT: true se lo stato è entangled
    bool isEntangled(const Matrix4cd& rho) {
        SelfAdjointEigenSolver<Matrix4cd> es(partialTranspose(rho));
        // Se l'autovalore minimo è negativo, lo stato è entangled
        return (es.eigenvalues().minCoeff() < -1e-12);
    }

    // Calcolo del valore massimo della disuguaglianza di Bell (CHSH)
    // Criterio di Horodecki: S = 2 * sqrt(u1 + u2) 
    double getBellValue(const Matrix4cd& rho) {
        Matrix3d T = Matrix3d::Zero();
        
        // Costruzione della matrice di correlazione T_ij = Tr(rho * sigma_i \otimes sigma_j)
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                Matrix4cd op = kroneckerProduct(pauli[i+1], pauli[j+1]);
                T(i, j) = (rho * op).trace().real();
            }
        }

        // Calcolo degli autovalori di M = T^T * T
        Matrix3d M = T.transpose() * T;
        SelfAdjointEigenSolver<Matrix3d> es(M);
        Vector3d u = es.eigenvalues(); // Ordinati in modo crescente u0, u1, u2

        // Bell è violato se S > 2.0
        return 2.0 * std::sqrt(u[2] + u[1]); 
    }

    // Get 32 feature RAW (real e imag) da rho
    std::vector<double> getRawFeatures(const Matrix4cd& rho) {
        std::vector<double> features;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                features.push_back(rho(i, j).real());
                features.push_back(rho(i, j).imag());
            }
        }
        return features;
    }
};

#endif