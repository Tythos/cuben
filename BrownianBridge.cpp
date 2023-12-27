/**
 * BrownianBridge.cpp
 */

#include "cuben.hpp"

cuben::BrownianBridge::BrownianBridge() {
    t0 = 1.0f;
    x0 = 1.0f;
    tf = 3.0f;
    xf = 2.0f;
}

cuben::BrownianBridge::BrownianBridge(float nt0, float nx0, float ntf, float nxf) {
    t0 = nt0;
    x0 = nx0;
    tf = ntf;
    xf = nxf;
}

Eigen::VectorXf cuben::BrownianBridge::getWalk(float dt) {
    Eigen::VectorXf ti = cuben::fundamentals::initRangeVec(t0, dt, tf);
    Eigen::VectorXf xi = Eigen::VectorXf(ti.rows());
    xi(0) = x0;
    Norm normPrng = Norm();
    for (int i = 1; i < ti.rows(); i++) {
        float f = (xf - xi(i-1)) / (tf - ti(i-1));
        xi(i) = xi(i-1) + f * (ti(i) - ti(i-1)) + normPrng.roll() * std::sqrt(ti(i) - ti(i-1));
    }
    return xi;
}

float cuben::BrownianBridge::bbf(float t, float x) {
    return (xf - x) / (tf - t);
}

float cuben::BrownianBridge::bbg(float t, float x) {
    return 1.0f;
}
