/**
 * tests/test_bvp.cpp
 */

#include "cuben.hpp"
#include "gtest/gtest.h"

void refBvp(float t, Eigen::VectorXf x, Eigen::VectorXf &dxdt) {
    dxdt = Eigen::VectorXf(2);
    dxdt(0) = x(1);
    dxdt(1) = 4.0f * x(0);
}

Eigen::VectorXf nonLinF(Eigen::VectorXf xi, float dt) {
    int n = xi.rows();
    Eigen::VectorXf fi(n);
    Eigen::Vector2f xBounds; xBounds << 1.0f,4.0f;
    for (int i = 0; i < n; i++) {
        if (i == 0) {
            fi(i) = xBounds(0) - (2.0f + dt * dt) * xi(i) + dt * dt * xi(i) * xi(i) + xi(i+1);
        } else if (i == n - 1) {
            fi(i) = xi(i-1) - (2.0f + dt * dt) * xi(i) + dt * dt * xi(i) * xi(i) + xBounds(1);
        } else {
            fi(i) = xi(i-1) - (2.0f + dt * dt) * xi(i) + dt * dt * xi(i) * xi(i) + xi(i+1);
        }
    }
    return fi;
}

Eigen::MatrixXf nonLinDfdx(Eigen::VectorXf xi, float dt) {
    int n = xi.rows();
    Eigen::MatrixXf dfdx = Eigen::MatrixXf::Zero(n,n);
    for (int i = 0; i < n; i++) {
        if (i > 0) {
            dfdx(i,i-1) = 1.0f;
        }
        if (i < n - 1) {
            dfdx(i,i+1) = 1.0f;
        }
        dfdx(i,i) = 2.0f * dt * dt * xi(i) - (2.0f + dt * dt);
    }
    return dfdx;
}

void simpleODE(float t, Eigen::VectorXf x, Eigen::VectorXf& dxdt) {
    dxdt(0) = x(1);
    dxdt(1) = -x(0);
}

Eigen::VectorXf f(Eigen::VectorXf xi, float t) {
    Eigen::VectorXf result(2);
    result(0) = xi(1);
    result(1) = -2.0 * xi(0);
    return result;
}

Eigen::MatrixXf dfdx(Eigen::VectorXf xi, float t) {
    Eigen::MatrixXf jacobian(2,2);
    jacobian(0,0) = 0;
    jacobian(0,1) = 1;
    jacobian(1,0) = -2.0;
    jacobian(1,1) = 0;
    return jacobian;
}

namespace cuben {
    namespace tests {
        namespace test_bvp {
            TEST(TestBvp, OldTest) {
                Eigen::Vector3f coeffs; coeffs << 0.0f,4.0f,0.0f;
                Eigen::Vector2f tBounds; tBounds << 0.0f,1.0f;
                Eigen::Vector2f xBounds; xBounds << 1.0f,3.0f;
                Eigen::VectorXf xi = cuben::bvp::bSplineGal(coeffs, tBounds, xBounds, 0.05f);
                Eigen::VectorXf xi_expected(20); xi_expected <<
                    1.0f, 0.992131f, 0.994199f, 1.00623f, 1.02833f, 1.06074f, 1.10377f, 1.15786f, 1.22355f, 1.30149f, 1.39247f, 1.49739f, 1.61732f, 1.75345f, 1.90714f, 2.07993f, 2.27356f, 2.48996f, 2.7313f, 3.0f;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(xi, xi_expected, 1e-3, true));
            }

            // TEST(TestBvp, ShooterTest) {
            //     Eigen::Vector2f tBounds(0.0f, 3.14159f);
            //     Eigen::Vector2f xBounds(0.0f, 0.0f);
            //     Eigen::VectorXf ti;
            //     Eigen::MatrixXf xi;
            //     cuben::bvp::shoot(simpleODE, tBounds, xBounds, ti, xi);
            //     std::cout << ti << std::endl;
            //     std::cout << xi << std::endl;
            //     ASSERT_NEAR(xi(xi.rows()-1, 0), xBounds(1), 1e-4);
            // }

            // TEST(TestBvp, LtiFinElTest) {
            //     float dt = 0.1;
            //     Eigen::Vector3f coeffs; coeffs <<
            //         1.0, 0.0, 0.0;
            //     Eigen::Vector2f tBounds(0.0, 10.0);
            //     Eigen::Vector2f xBounds(0.0, 10.0);
            //     Eigen::VectorXf ti;
            //     Eigen::VectorXf xi;
            //     cuben::bvp::ltiFinEl(coeffs, tBounds, xBounds, dt, ti, xi);
            //     std::cout << ti << std::endl;
            //     std::cout << xi << std::endl;
            //     ASSERT_TRUE(true);
            // }

            // TEST(TestBvp, NonLinFinElTest) {
            //     float dt = 0.1;
            //     Eigen::Vector2f tBounds(0.0, 1.0);
            //     Eigen::Vector2f xBounds(0.0, 1.0);
            //     Eigen::VectorXf ti, xi;
            //     cuben::bvp::nonLinFinEl(f, dfdx, tBounds, xBounds, dt, ti, xi);
            //     std::cout << ti << std::endl;
            //     std::cout << xi << std::endl;
            // }

            TEST(TestBvp, ColloPolyTest) {
                Eigen::Vector3f coeffs(1.0f, 0.0f, -1.0f);
                Eigen::Vector2f tBounds(0.0f, 1.0f);
                Eigen::Vector2f xBounds(1.0f, 0.0f);
                float dt = 0.1f;
                Eigen::VectorXf resultCoeffs = cuben::bvp::colloPoly(coeffs, tBounds, xBounds, dt);
                Eigen::VectorXf expectedCoeffs(10); expectedCoeffs <<
                    1.0f, -21.8575f, 0.500018f, -3.64306f, 0.0422948f, -0.183831f, 0.00414906f, -0.00704313f, 0.00148262f, -0.000392208f;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(resultCoeffs, expectedCoeffs, 1e-3, true));
            }

            TEST(TestBvp, ColloChebyTest) {
                Eigen::Vector4f coeffs; coeffs <<
                    1.0f, 0.0f, -1.0f, 0.0f;
                Eigen::Vector2f tBounds(0, 1);
                Eigen::Vector2f xBounds(1, std::exp(1.0f) + 1.0f/std::exp(1.0f));
                int n = 10;
                Eigen::VectorXf result = cuben::bvp::colloCheby(coeffs, tBounds, xBounds, n);
                Eigen::VectorXf expected(n); expected <<
                    1.92595f, 1.02231f, 0.116331f, 0.0206902f, 0.00079923f, 7.5611e-05f, 1.84263e-06f, 1.20691e-07f, 2.07632e-09f, 1.14142e-10f;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(result, expected, 1e-3, true));
            }
        }
    }
}

int main(int nArgs, char** vArgs) {
    ::testing::InitGoogleTest(&nArgs, vArgs);
    return RUN_ALL_TESTS();
}
