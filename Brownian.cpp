/**
 * Brownian.cpp
 */

#include "cuben.hpp"

cuben::Brownian::Brownian() {
    normPrng = cuben::Norm();
}

Eigen::VectorXf cuben::Brownian::sampleWalk(Eigen::VectorXf xi) {
    Eigen::VectorXf yi = Eigen::VectorXf(xi.rows());
    yi(0) = 0.0f;
    for (int i = 1; i < xi.rows(); i++) {
        yi(i) = yi(i-1) + std::sqrt(xi(i) - xi(i-1)) * normPrng.roll();
    }
    return yi;
}
