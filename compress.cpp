/**
 * compress.cpp
 */

#include "cuben.hpp"

Eigen::VectorXf cuben::compress::dct(Eigen::VectorXf xi) {
    unsigned int n = xi.rows();
    float sqrt2 = std::sqrt(2.0f);
    float c = sqrt2 / std::sqrt((float)n);
    Eigen::MatrixXf C = Eigen::MatrixXf(n,n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == 0) {
                C(i,j) = c * 0.5f * sqrt2;
            } else {
                C(i,j) = c * std::cos(((float)i * ((float)j + 0.5f) * M_PI) / (float)n);
            }
        }
    }
    return C * xi;
}

float cuben::compress::dctInterp(Eigen::VectorXf Xi, float t) {
    unsigned int n = Xi.rows();
    float invSqrtN = 1.0f / std::sqrt((float)n);
    float c = std::sqrt(2) * invSqrtN;
    float y = invSqrtN * Xi(0);
    for (int i = 0; i < n; i++) {
        y += c * Xi(i) * std::cos(0.5f * (float)i * (2.0f * t + 1.0f) * M_PI / (float)n);
    }
    return y;
}

Eigen::VectorXf cuben::compress::dctFit(Eigen::VectorXi xi, unsigned int n) {
    const unsigned int N = xi.size();
    Eigen::VectorXf xf(N);
    for (int i = 0; i < N; i += 1) {
        xf[i] = (float)xi[i];
    }
    Eigen::VectorXf Xf = cuben::compress::dct(xf);
    return Xf.segment(0,n);
}

Eigen::MatrixXf cuben::compress::dct2d(Eigen::MatrixXf xij) {
    unsigned int n = xij.rows();
    float sqrtInvN = std::sqrt(1.0f / (float)n);
    float sqrt2 = std::sqrt(2.0f);
    if (n != xij.cols()) {
        throw cuben::exceptions::xMismatchedDims();
    }
    Eigen::MatrixXf C = Eigen::MatrixXf(n,n);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == 0) {
                C(i,j) = sqrtInvN;
            } else {
                C(i,j) = sqrt2 * sqrtInvN * std::cos(i * (j + 0.5f) * M_PI / (float)n);
            }
        }
    }
    return C * xij * C.transpose();
}

Eigen::MatrixXf cuben::compress::idct2d(Eigen::MatrixXf Xij) {
    unsigned int n = Xij.rows();
    float sqrtInvN = std::sqrt(1.0f / (float)n);
    float sqrt2 = std::sqrt(2.0f);
    if (n != Xij.cols()) {
        throw cuben::exceptions::xMismatchedDims();
    }
    Eigen::MatrixXf C = Eigen::MatrixXf(n,n);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == 0) {
                C(i,j) = sqrtInvN * std::cos(0.5f * i * (2.0f * j + 1.0f) * M_PI / (float)n);
            } else {
                C(i,j) = sqrt2 * sqrtInvN * std::cos(0.5f * i * (2.0f * j + 1.0f) * M_PI / (float)n);
            }
        }
    }
    return C.transpose() * Xij * C;
}

Eigen::MatrixXi cuben::compress::applyQuant(Eigen::MatrixXf Xij, Eigen::MatrixXf Qij) {
    // Compresses the given matrix by applying quantization defined by Qij
    unsigned int n = Xij.rows();
    Eigen::MatrixXi Yij = Eigen::MatrixXi(n,n);
    if (n != Xij.cols() || n != Qij.rows() || n != Qij.cols()) {
        throw cuben::exceptions::xMismatchedDims();
    }
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Yij(i,j) = (int)(Xij(i,j) / Qij(i,j));
        }
    }
    return Yij;
}

Eigen::MatrixXf cuben::compress::reverseQuant(Eigen::MatrixXi Yij, Eigen::MatrixXf Qij) {
    // Decompress the given matrix by reversing quantization defined by Qij
    unsigned int n = Yij.rows();
    Eigen::MatrixXf Xij = Eigen::MatrixXf(n,n);
    if (n != Yij.cols() || n != Qij.rows() || n != Qij.cols()) {
        throw cuben::exceptions::xMismatchedDims();
    }
    
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            Xij(i,j) = (float)Yij(i,j) * Qij(i,j);
        }
    }
    return Xij;
}

Eigen::MatrixXf cuben::compress::linearQuant(unsigned int n, float p) {
    // Constructs a linear quantization matrix of size n x n; compression
    // is controlled by the parameter p (greater p = greater compression)
    Eigen::MatrixXf Qij = Eigen::MatrixXf(n,n);
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            Qij(i,j) = 8.0f * p * ((float)i + (float)j + 1.0f);
        }
    }
    return Qij;
}

Eigen::MatrixXf cuben::compress::jpegQuant(float p) {
    Eigen::MatrixXf Qij = Eigen::MatrixXf(8,8);
    Qij.row(0) << 16.0f,11.0f,10.0f,16.0f,24.0f,40.0f,51.0f,61.0f;
    Qij.row(1) << 12.0f,12.0f,14.0f,19.0f,26.0f,58.0f,60.0f,55.0f;
    Qij.row(2) << 14.0f,13.0f,16.0f,24.0f,40.0f,57.0f,69.0f,56.0f;
    Qij.row(3) << 14.0f,17.0f,22.0f,29.0f,51.0f,87.0f,80.0f,62.0f;
    Qij.row(4) << 18.0f,22.0f,37.0f,56.0f,68.0f,109.0f,103.0f,77.0f;
    Qij.row(5) << 24.0f,35.0f,55.0f,64.0f,81.0f,104.0f,113.0f,92.0f;
    Qij.row(6) << 49.0f,64.0f,78.0f,87.0f,103.0f,121.0f,120.0f,101.0f;
    Qij.row(7) << 72.0f,92.0f,95.0f,98.0f,112.0f,100.0f,103.0f,99.0f;
    return p * Qij;
}

Eigen::MatrixXf cuben::compress::baseYuvQuant() {
    Eigen::MatrixXf Qij = Eigen::MatrixXf(8,8);
    Qij.row(0) << 17.0f,18.0f,24.0f,47.0f,99.0f,99.0f,99.0f,99.0f;
    Qij.row(1) << 18.0f,21.0f,26.0f,66.0f,99.0f,99.0f,99.0f,99.0f;
    Qij.row(2) << 24.0f,26.0f,56.0f,99.0f,99.0f,99.0f,99.0f,99.0f;
    Qij.row(3) << 47.0f,66.0f,99.0f,99.0f,99.0f,99.0f,99.0f,99.0f;
    Qij.row(4) << 99.0f,99.0f,99.0f,99.0f,99.0f,99.0f,99.0f,99.0f;
    Qij.row(5) << 99.0f,99.0f,99.0f,99.0f,99.0f,99.0f,99.0f,99.0f;
    Qij.row(6) << 99.0f,99.0f,99.0f,99.0f,99.0f,99.0f,99.0f,99.0f;
    Qij.row(7) << 99.0f,99.0f,99.0f,99.0f,99.0f,99.0f,99.0f,99.0f;
    return Qij;
}
