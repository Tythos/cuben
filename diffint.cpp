/**
 * diffint.cpp
 */

#include "cuben.hpp"

float cuben::diffint::dfdx_2pfd(float(*f)(float), float x, float h) {
    return (f(x + h) - f(x)) / h;
}

float cuben::diffint::dfdx_3pcd(float(*f)(float), float x, float h) {
    return (f(x + h) - f(x - h)) / (2 * h);
}

float cuben::diffint::d2fdx2_3pcd(float(*f)(float), float x, float h) {
    return (f(x - h) - 2 * f(x) + f(x + h)) / (h * h);
}

float cuben::diffint::dfdx_5pcd(float(*f)(float), float x, float h) {
    return (f(x - h) - 8 * f(x - 0.5 * h) + 8 * f(x + 0.5 * h) - f(x + h)) / (6 * h);
}

float cuben::diffint::d2fdx2_5pcd(float(*f)(float), float x, float h) {
    return (-f(x - h) + 16 * f(x - 0.5 * h) - 30 * f(x) + 16 * f(x + 0.5 * h) - f(x + h)) / (3 * h * h);
}

float cuben::diffint::dfdx_cubic(float(*f)(float), float x, float h) {
    Eigen::VectorXf xi(4); xi << x - 1.5 * h, x - 0.5 * h, x + 0.5 * h, x + 1.5 * h;
    Eigen::VectorXf b(4); b << f(xi(0)), f(xi(1)), f(xi(2)), f(xi(3));
    Eigen::MatrixXf A(4,4); A << xi(0) * xi(0) * xi(0), xi(0) * xi(0), xi(0), 1.0f, xi(1) * xi(1) * xi(1), xi(1) * xi(1), xi(1), 1.0f, xi(2) * xi(2) * xi(2), xi(2) * xi(2), xi(2), 1.0f, xi(3) * xi(3) * xi(3), xi(3) * xi(3), xi(3), 1.0f;
    Eigen::VectorXf c = cuben::systems::paluSolve(A, b);
    return 3 * c(0) * x * x + 2 * c(1) * x + c(2);
}

float cuben::diffint::intfdx_trap(float(*f)(float), float x0, float x1) {
    float h = x1 - x0;
    return 0.5 * h * (f(x0) + f(x1));
}

float cuben::diffint::intfdx_simp(float(*f)(float), float x0, float x2) {
    float x1 = 0.5 * (x0 + x2);
    float h = 0.5 * (x2 - x0);
    return h * (f(x0) + 4 * f(x1) + f(x2)) / 3.0f;
}

float cuben::diffint::intfdx_simp38(float(*f)(float), float x0, float x3) {
    float x1 = x0 + (x3 - x0) / 3;
    float x2 = x0 + 2 * (x3 - x0) / 3;
    float h = x1 - x0;
    return 3 * h * (f(x0) + 3 * f(x1) + 3 * f(x2) + f(x3)) / 8;
}

float cuben::diffint::intfdx_mid(float(*f)(float), float x0, float x1) {
    return (x1 - x0) * f(0.5 * (x0 + x1));
}

float cuben::diffint::intfdx_compTrap(float(*f)(float), float x0, float x1, int n) {
    float yNet = f(x0) + f(x1);
    float dx = (x1 - x0) / n;
    for (int i = 1; i < n; i++) {
        yNet += 2 * f(x0 + i * dx);
    }
    return dx * yNet / 2;
}

float cuben::diffint::intfdx_compSimp(float(*f)(float), float x0, float x1, int n) {
    float yNet = f(x0) + f(x1);
    float dx = (x1 - x0) / n;
    for (int i = 0; i < n - 1; i++) {
        yNet += 4 * f(x0 + (i + 0.5) * dx) + 2 * f(x0 + (i + 1) * dx);
    }
    yNet += 4 * f(x0 + (n - 0.5) * dx);
    return dx * yNet / 6;
}

float cuben::diffint::intfdx_compMid(float(*f)(float), float x0, float x1, int n) {
    float yNet = 0.0f;
    float dx = (x1 - x0) / n;
    for (int i = 0; i < n; i++) {
        yNet += f(x0 + (i + 0.5) * dx);
    }
    return dx * yNet;
}

float cuben::diffint::intfdx_romberg(float(*f)(float), float x0, float x1, int n) {
    Eigen::MatrixXf R(n,n);
    float h = 0.0f;
    R(0,0) = 0.5f * (x1 - x0) * (f(x0) + f(x1));
    for (int i = 1; i < n; i++) {
        h = (x1 - x0) / std::pow(2, i);
        R(i,0) = 0.0f;
        for (int j = 0; j < std::pow(2,i-1); j++) {
            R(i,0) += f(x0 + (2 * j + 1) * h);
        }
        R(i,0) = 0.5 * R(i-1,0) + h * R(i,0);
        for (int j = 1; j <= i; j++) {
            R(i,j) = (std::pow(4,j) * R(i,j-1) - R(i-1,j-1)) / (std::pow(4,j) - 1);
        }
    }
    return R(n-1,n-1);
}

float cuben::diffint::intfdx_adaptTrap(float(*f)(float), float x0, float x1, float tol) {
    float xMid = 0.5f * (x0 + x1);
    float appxAll = 0.5 * (x1 - x0) * (f(x0) + f(x1));
    float appxLeft = 0.5 * (xMid - x0) * (f(x0) + f(xMid));
    float appxRight = 0.5 * (x1 - xMid) * (f(xMid) + f(x1));
    float error = std::abs(appxAll - appxLeft - appxRight);
    float result = 0.0f;
    if (error < 3 * tol) {
        result = appxLeft + appxRight;
    } else {
        result = cuben::diffint::intfdx_adaptTrap(f, x0, xMid, 0.5f * tol) + cuben::diffint::intfdx_adaptTrap(f, xMid, x1, 0.5f * tol);
    }
    return result;
}

float cuben::diffint::intfdx_adaptSimp(float(*f)(float), float x0, float x1, float tol) {
    float xMid = 0.5f * (x0 + x1);
    float dx = 0.5f * (x1 - x0);
    float appxAll = dx * (f(x0) + 4 * f(xMid) + f(x1)) / 3.0f;
    float appxLeft = 0.5f * dx * (f(x0) + 4 * f(x0 + 0.5f * dx) + f(xMid)) / 3.0f;
    float appxRight = 0.5f * dx * (f(xMid) + 4 * f(xMid + 0.5f * dx) + f(x1)) / 3.0f;
    float error = std::abs(appxAll - appxLeft - appxRight);
    float result = 0.0f;
    if (error < 15 * tol) {
        result = appxLeft + appxRight;
    } else {
        result = cuben::diffint::intfdx_adaptSimp(f, x0, xMid, 0.5f * tol) + cuben::diffint::intfdx_adaptSimp(f, xMid, x1, 0.5f * tol);
    }
    return result;
}

float cuben::diffint::intfdx_gaussQuad(float(*f)(float), float x0, float x1, int n) {
    float sum = 0.0f;
    if (n == 2) {
        float il = -std::sqrt(1.0f / 3.0f);
        float ir = std::sqrt(1.0f / 3.0f);
        float xl = x0 + 0.5f * (x1 - x0) * (il + 1);
        float xr = x0 + 0.5f * (x1 - x0) * (ir + 1);
        sum = 0.5f * (x1 - x0) * (1.0f * f(xl) + 1.0f * f(xr));
    } else if (n == 3) {
        float il = -std::sqrt(3.0f / 5.0f);
        float im = 0.0f;
        float ir = std::sqrt(3.0f / 5.0f);
        float xl = x0 + 0.5f * (x1 - x0) * (il + 1);
        float xm = x0 + 0.5f * (x1 - x0) * (im + 1);
        float xr = x0 + 0.5f * (x1 - x0) * (ir + 1);
        sum = 0.5f * (x1 - x0) * ((5.0f / 9.0f) * f(xl) + (8.0f / 9.0f) * f(xm) + (5.0f / 9.0f) * f(xr));
    } else if (n == 4) {
        float ill = -std::sqrt((15 + 2 * std::sqrt(30)) / 35);
        float il = -std::sqrt((15 - 2 * std::sqrt(30)) / 35);
        float ir = std::sqrt((15 - 2 * std::sqrt(30)) / 35);
        float irr = std::sqrt((15 + 2 * std::sqrt(30)) / 35);
        float xll = x0 + (x1 - x0) * 0.5f * (ill + 1);
        float xl = x0 + (x1 - x0) * 0.5f * (il + 1);
        float xr = x0 + (x1 - x0) * 0.5f * (ir + 1);
        float xrr = x0 + (x1 - x0) * 0.5f * (irr + 1);
        sum = 0.5f * (x1 - x0) * (((90 - 5 * std::sqrt(30.0f)) / 180) * f(xll) + ((90 + 5 * std::sqrt(30.0f)) / 180) * f(xl) + ((90 + 5 * std::sqrt(30.0f)) / 180) * f(xr) + ((90 - 5 * std::sqrt(30.0f)) / 180) * f(xrr));
    }
    return sum;
}
