/**
 * polynomial.cpp
 */

#include "polynomial.hpp"

cuben::polynomial::Polynomial::Polynomial() {
    ri = Eigen::VectorXf();
    ci = Eigen::VectorXf();
}
		
void cuben::polynomial::Polynomial::print() {
    int n = ci.rows();
    std::cout << "P(x) = ";
    if (n == 0) {
        std::cout << "?" << std::endl;
    } else {
        for (int i = n - 1; i >= 0; i--) {
            if (i > 0) {
                std::cout << ci(i) << " + ";
            } else {
                std::cout << ci(i);
            }
            if (i > 0) {
                if (ri(i) == 0) {
                    std::cout << "x * ";
                } else {
                    std::cout << "(x - " << ri(i) << ") * ";
                }
                if (i > 1) std::cout << "( ";
            }
        }
        for (int i = n - 1; i > 1; i--) {
            std::cout << " )";
        }
        std::cout << std::endl;
    }
}
		
float cuben::polynomial::Polynomial::eval(float x) {
    int n = ci.rows();
    if (n == 0) { return 0; }
    float y = ci(0);
    for (int i = 1; i < n; i++) {
        y = y * (x - ri(i)) + ci(i);
    }
    return y;
}

void cuben::polynomial::Polynomial::push(float r, float c) {
    int n = ci.rows();
    ri.conservativeResize(n + 1);
    ci.conservativeResize(n + 1);
    ri(n) = r;
    ci(n) = c;
}

int cuben::polynomial::Polynomial::size() {
    return ci.rows();
}