/**
 * interp.cpp
 */

#include "cuben.hpp"

float cuben::interp::lagrange(Eigen::VectorXf xi, Eigen::VectorXf yi, float x) {
    int n = xi.rows();
    if (n != yi.rows()) {
        throw cuben::exceptions::xMismatchedPoints();
    }
    Eigen::VectorXf t(n);
    float num, denom;
    for (int i = 0; i < n; i++) {
        num = denom = 1.0f;
        for (int j = 0; j < n; j++) {
            if (i != j) {
                num *= x - xi(j);
                denom *= xi(i) - xi(j);
            }
        }
        t(i) = yi(i) * num / denom;
    }
    return t.sum();
}

float cuben::interp::sinInterp(float x) {
    // deliberately erroneous test to illustrate poor interpolation sampling of SIN(x)
    cuben::Polynomial p;
    if (p.getNumPoints() == 0) {
        p.push(0.0f, 0.0f);
        p.push(M_PI / 6, 0.5f);
        p.push(2 * M_PI / 6, 0.866f);
        p.push(3 * M_PI / 6, 1.0f);
    }
    float xEquiv = fmod(x, 2 * M_PI);
    if (xEquiv < 0.5 * M_PI) {
        return p.eval(xEquiv);
    } else if (xEquiv < M_PI) {
        return p.eval(M_PI - xEquiv);
    } else if (xEquiv < 1.5 * M_PI) {
        return -p.eval(xEquiv - M_PI);
    } else {
        return -p.eval(2 * M_PI - xEquiv);
    }    
}

cuben::Polynomial cuben::interp::chebyshev(float(*f)(float), float xMin, float xMax, int n) {
    cuben::Polynomial p;
    float x;
    for (int i = 0; i < n; i++) {
        x = 0.5 * (xMin + xMax) + 0.5 * (xMax - xMin) * std::cos((2 * (i - 1) - 1) * M_PI / (2 * n));
        p.push(x, f(x));
    }
    return p;
}

Eigen::VectorXf cuben::interp::cheb(Eigen::VectorXf t, int order) {
    int n = t.rows();
    if (order == 0) {
        return Eigen::VectorXf::Ones(n);
    } else if (order == 1) {
        return Eigen::VectorXf(t);
    } else {
        return 2.0f * t.cwiseProduct(cuben::interp::cheb(t, order - 1)) - cuben::interp::cheb(t, order - 2);
    }
}

Eigen::VectorXf cuben::interp::dchebdt(Eigen::VectorXf t, int order) {
    int n = t.rows();
    if (order == 0) {
        return Eigen::VectorXf::Zero(n);
    } else if (order == 1) {
        return Eigen::VectorXf::Ones(n);
    } else if (order == 2) {
        return 4.0f * Eigen::VectorXf(t);
    } else {
        return 2.0f * cuben::interp::cheb(t, order - 1) + 2.0f * t.cwiseProduct(cuben::interp::dchebdt(t, order - 1)) - cuben::interp::dchebdt(t, order - 2);
    }
}

Eigen::VectorXf cuben::interp::d2chebdt2(Eigen::VectorXf t, int order) {
    int n = t.rows();
    if (order == 0) {
        return Eigen::VectorXf::Zero(n);
    } else if (order == 1) {
        return Eigen::VectorXf::Zero(n);
    } else if (order == 2) {
        return 4.0f * Eigen::VectorXf::Ones(n);
    } else if (order == 3) {
        return 12.0f * t;
    } else {
        return 4.0f * cuben::interp::dchebdt(t, order - 1) + 2 * t.cwiseProduct(cuben::interp::d2chebdt2(t, order - 1)) - cuben::interp::d2chebdt2(t, order - 2);
    }
}

Eigen::VectorXf cuben::interp::chebSamp(float lhs, int n, float rhs) {
    // Return array of points sampled between [lhs,rhs] using a Chebyshev distribution
    Eigen::VectorXf xi(n);
    for (int i = 0; i < n; i++) {
        xi(i) = lhs + (rhs - lhs) * (0.5f - 0.5f * std::cos(M_PI * i / (n - 1)));
    }
    return xi;
}
