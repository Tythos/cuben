/**
 * tests/test_ode.cpp
 */

#include "cuben.hpp"
#include "gtest/gtest.h"

float dxdtSng(float t, float x) {
    return t * x + t * t * t;
}

float dxdtStiff(float t, float x) {
    return x + 8.0f * x * x - 9.0f * x * x * x;
}

float dfdxStiff(float t, float x) {
    return 1.0f + 16.0f * x - 27.0f * x * x;
}

void pendSys(float t, Eigen::VectorXf x, Eigen::VectorXf &result) {
    float g = 9.81;
    float l = 1.0f;
    result(0) = x(1);
    result(1) = -g * std::sin(x(0)) / l;
}

void scalarSys(float t, Eigen::VectorXf x, Eigen::VectorXf &result) {
    result(0) = t * x(0) + t * t * t;
}

float dxdtPatho(float t, float x) {
    return -4 * t * t * t * x * x;
}

void dxdtVec(float t, Eigen::VectorXf xi, Eigen::VectorXf& dxidt) {
    dxidt(0) = t * xi(0) + t * t * t;
}

void test_dxdt(float t, Eigen::VectorXf x, Eigen::VectorXf& dx) {
    dx(0) = -2 * x(0); // dx/dt = -2x
}

void harmonic_motion(float t, Eigen::VectorXf x, Eigen::VectorXf& dx) {
    dx(0) = x(1);
    dx(1) = -x(0);
}

float dxdt_function(float t, float x) {
    return x;
}

float d2xdt2_function(float t, float x) {
    return x;
}

float d2xdtdx_function(float t, float x) {
    return 1.0f;
}

void system_function(float t, Eigen::VectorXf x, Eigen::VectorXf& dxdt) {
    dxdt(0) = x(1);
    dxdt(1) = -x(0);
}

void predatorPrey(float t, Eigen::VectorXf x, Eigen::VectorXf& dx) {
    const float a = 0.1f;
    const float b = 0.02f;
    const float c = 0.01f;
    const float d = 0.1f;
    dx(0) = a * x(0) - b * x(0) * x(1);
    dx(1) = c * x(0) * x(1) - d * x(1);
}

void vanDerPol(float t, Eigen::VectorXf x, Eigen::VectorXf& dx) {
    const float mu = 2.0f;
    dx(0) = x(1);
    dx(1) = mu * (1.0f - x(0) * x(0)) * x(1) - x(0);
}

float fInd(float t, float x) {
    return -x;
}

float fImp(float t, float x) {
    return -1.0f;
}

float another_test_dxdt(float t, float x) {
    return -0.1f * x;
}

float simpleODE(float t, float x) {
    (void)t;
    return 2.0f * x;
}

float fIndTest(float t, float x) {
    return t * x;
}

float fImpTest(float t, float x) {
    return t + x;
}

float yet_another_test(float t, float x) {
    return -0.5f * x;
}

float harmonicOscillator(float t, float x) {
    const float zeta = 0.5f;
    const float omega = 2.0f;
    return -2.0f * zeta * omega * x - omega * omega * x;
}

namespace cuben {
    namespace test {
        namespace test_ode {
            TEST(TestOde, OldPendTest) {
                Eigen::Vector2f tInt; tInt << 0.0f,1.0f;
                Eigen::VectorXf x0(2); x0 << M_PI/2.0f,0.0f;
                Eigen::VectorXf ti; Eigen::MatrixXf xi(0,0);
                cuben::ode::rk23Sys(pendSys, tInt, x0, ti, xi);
                std::cout << "t\tx0\tx1" << std::endl;
                for (int i = 0; i < 10; i += 1) {
                    std::cout << ti(i) << "\t" << xi(i,0) << "\t" << xi(i,1) << std::endl;
                }
                Eigen::VectorXf ti_ref(10); ti_ref <<
                    0.0, 0.000976562, 0.00219727, 0.00372314, 0.00563049, 0.00682259, 0.0083127, 0.0101753, 0.0113395, 0.0127947;
                Eigen::VectorXf xi0_ref(10); xi0_ref <<
                    1.5708, 1.57079, 1.57076, 1.57071, 1.57061, 1.57054, 1.57042, 1.57024, 1.57011, 1.56993;
                Eigen::VectorXf xi1_ref(10); xi1_ref <<
                    0, -0.00958008, -0.0215552, -0.036524, -0.0552351, -0.0669296, -0.0815476, -0.0998202, -0.11124, -0.125516;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(ti.segment(0, 10), ti_ref, 1e-3, true));
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(xi.col(0).segment(0, 10), xi0_ref, 1e-3, true));
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(xi.col(1).segment(0, 10), xi1_ref, 1e-3, true));
            }

            TEST(TestOde, OldEulerTest) {
                Eigen::VectorXf ti, xi;
                std::cout << std::endl << "Example 6.2, Euler's Method of ty+t^3 on [0,1], y0=0, h=0.1:" << std::endl;
                ti = cuben::fundamentals::initRangeVec(0.0,0.1,1.0);
                xi = cuben::ode::euler(dxdtSng, ti, 1.0);
                cuben::fundamentals::printVecSeries(ti, xi, "Time", "Value");
                Eigen::VectorXf ti_ref(10); ti_ref <<
                    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9;
                Eigen::VectorXf xi_ref(10); xi_ref << 
                    1.0, 1.0, 1.0101, 1.0311, 1.06474, 1.11372, 1.18191, 1.27443, 1.39794, 1.56097;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(ti, ti_ref, 1e-3, true));
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(xi, xi_ref, 1e-3, true));
            }

            TEST(TestOde, OldEulerPatho) {
                Eigen::VectorXf ti, xi;
                float dt;
                std::cout << std::endl << "Example 6.9, Euler's Method of -4t3y2 on [-10,0], y0=1/10001, h=1e-[3,4,5]:" << std::endl;
                std::cout << "Expected f(0): " << 1.0 << std::endl;
                dt = 1e-3;
                ti = cuben::fundamentals::initRangeVec(-10.0, dt, 0.0);
                xi = cuben::ode::euler(dxdtPatho, ti, 1.0/10001.0);
                std::cout << "Actual, for step size " << dt << ": " << xi(xi.rows()-1) << std::endl;
                ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(xi(xi.rows()-1), 0.233635, 1e-3, true));
                dt = 1e-4;
                ti = cuben::fundamentals::initRangeVec(-10.0, dt, 0.0);
                xi = cuben::ode::euler(dxdtPatho, ti, 1.0/10001.0);
                std::cout << "Actual, for step size " << dt << ": " << xi(xi.rows()-1) << std::endl;
                ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(xi(xi.rows()-1), 0.745913, 1e-3, true));
                dt = 1e-5;
                ti = cuben::fundamentals::initRangeVec(-10.0, dt, 0.0);
                xi = cuben::ode::euler(dxdtPatho, ti, 1.0/10001.0);
                std::cout << "Actual, for step size " << dt << ": " << xi(xi.rows()-1) << std::endl;                
                ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(xi(xi.rows()-1), 0.848635, 1e-3, true));
            }

            TEST(TestOde, OldTrapTest) {
                Eigen::VectorXf ti, xi;
                std::cout << std::endl << "Example 6.10, Trapezoid Method of ty+t^3 on [0,1], y0=0, h=0.1:" << std::endl;
                ti = cuben::fundamentals::initRangeVec(0.0,0.1,1.0);
                xi = cuben::ode::trap(dxdtSng, ti, 1.0);
                Eigen::VectorXf ti_ref(10); ti_ref <<
                    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9;
                Eigen::VectorXf xi_ref(10); xi_ref <<
                    1.0, 1.00505, 1.02068, 1.04826, 1.09018, 1.14994, 1.23234, 1.34374, 1.4924, 1.68898;
                cuben::fundamentals::printVecSeries(ti, xi, "Time", "Value");
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(ti, ti_ref, 1e-3, true));
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(xi, xi_ref, 1e-3, true));
            }

            TEST(TestOde, OldTrapPatho) {
                Eigen::VectorXf ti, xi;
                float dt;
                std::cout << std::endl << "Example 6.11, Trapeoid Method of -4t3y2 on [-10,0], y0=1/10001, h=1e-[3,4,5]:" << std::endl;
                std::cout << "Expected f(0): " << 1.0 << std::endl;
                dt = 1.5e-3;
                ti = cuben::fundamentals::initRangeVec(-10.0, dt, 0.0);
                xi = cuben::ode::trap(dxdtPatho, ti, 1.0/10001.0);
                std::cout << "Actual, for step size " << dt << ": " << xi(xi.rows()-1) << std::endl;
                ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(xi(xi.rows()-1), 1.001, 1e-3, true));
            }

            TEST(TestOde, OldRk4Test) {
                Eigen::VectorXf ti, xi;
                std::cout << std::endl << "Example 6.18, Runge-Kutta 4 of ty+t3 on [0,1], y(0)=1, h=0.1:" << std::endl;
                ti = cuben::fundamentals::initRangeVec(0.0,0.1,1.0);
                xi = cuben::ode::rk4(dxdtSng, ti, 1.0);
                cuben::fundamentals::printVecSeries(ti, xi, "Time", "Value");
                Eigen::VectorXf ti_ref(10); ti_ref <<
                    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9;
                Eigen::VectorXf xi_ref(10); xi_ref <<
                    1.0, 1.00504, 1.0206, 1.04808, 1.08986, 1.14945, 1.23165, 1.34286, 1.49138, 1.68791;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(ti, ti_ref, 1e-3, true));
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(xi, xi_ref, 1e-3, true));
            }

            //TEST(TestOde, OldRk45Test) {
            //    Eigen::VectorXf ti, xi;
            //    Eigen::Vector2f tInt;
            //    Eigen::MatrixXf xij;
            //    std::cout << std::endl << "Example 6.22, Runge-Kutta 4/5 of ty+t3 on [0,1], y(0)=1:" << std::endl;
            //    tInt << 0.0, 1.0;
            //    xi = cuben::fundamentals::safeResize(xi, 1);
            //    cuben::ode::rk45Sys(dxdtVec, tInt, xi, ti, xij);
            //    cuben::fundamentals::printVecSeries(ti, xij.col(0), "Time", "Value");
            //}

            TEST(TestOde, BasicRk45Test) {
                Eigen::Vector2f tInt(0.0f, 2.0f);
                Eigen::VectorXf x0(1);
                x0(0) = 1.0f;
                Eigen::VectorXf ti;
                Eigen::MatrixXf xi;
                cuben::ode::rk45Sys(test_dxdt, tInt, x0, ti, xi);
                ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(xi.row(xi.rows() - 1)(0), 0.0188659, 1e-3, true));
            }

            TEST(TestOde, MultiRk45Test) {
                Eigen::Vector2f tInt(0.0f, 2.0f);
                Eigen::VectorXf x0(2);
                x0(0) = 1.0f;
                x0(1) = 0.0f;
                Eigen::VectorXf ti;
                Eigen::MatrixXf xi;
                cuben::ode::rk45Sys(harmonic_motion, tInt, x0, ti, xi);
                Eigen::VectorXf expected(2); expected <<
                    -0.402635,
                    -0.915375;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(xi.row(xi.rows() - 1), expected, 1e-3, true));
            }

            TEST(TestOde, Taylor2ndTest) {
                Eigen::VectorXf ti(3); ti <<
                    0.0f, 1.0f, 2.0f;
                float x0 = 1.0f;
                Eigen::VectorXf xi = cuben::ode::taylor2nd(dxdt_function, d2xdt2_function, d2xdtdx_function, ti, x0);
                Eigen::VectorXf expected(3); expected <<
                    1.0, 3.0, 9.0;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(xi, expected, 1e-3, true));
            }

            TEST(TestOde, EulerSysTest) {
                Eigen::VectorXf ti(3); ti <<
                    0.0f, 1.0f, 2.0f;
                Eigen::VectorXf x0(2); x0 <<
                    1.0f, 0.0f;
                Eigen::MatrixXf xi = cuben::ode::eulerSys(system_function, ti, x0);
                Eigen::MatrixXf expected(3,2); expected <<
                    1.0f, 0.0f,
                    1.0f, -1.0f,
                    0.0f, -2.0f;
                ASSERT_TRUE(cuben::fundamentals::isMatrixWithinReltol(xi, expected, 1e-3, true));
            }

            TEST(TestOde, TrapSysTest) {
                Eigen::VectorXf ti(3); ti <<
                    0.0f, 1.0f, 2.0f;
                Eigen::VectorXf x0(2); x0 <<
                    1.0f, 0.0f;
                Eigen::MatrixXf xi = cuben::ode::trapSys(system_function, ti, x0);
                Eigen::MatrixXf expected(3,2); expected <<
                    1.0f, 0.0f,
                    0.5f, -1.0f,
                    -0.75f, -1.0f;
                ASSERT_TRUE(cuben::fundamentals::isMatrixWithinReltol(xi, expected, 1e-3, true));
            }

            TEST(TestOde, MidSysTest) {
                Eigen::VectorXf ti(3); ti <<
                    0.0f, 1.0f, 2.0f;
                Eigen::VectorXf x0(2); x0 <<
                    1.0f, 0.0f;
                Eigen::MatrixXf xi = cuben::ode::midSys(system_function, ti, x0);
                Eigen::MatrixXf expected(3,2); expected <<
                    1.0f, 0.0f,
                    0.5f, -1.0f,
                    -0.75f, -1.0f;
                ASSERT_TRUE(cuben::fundamentals::isMatrixWithinReltol(xi, expected, 1e-3, true));
            }

            TEST(TestOde, Rk4SysTest) {
                Eigen::VectorXf ti(3); ti <<
                    0.0f, 1.0f, 2.0f;
                Eigen::VectorXf x0(2); x0 <<
                    1.0f, 0.0f;
                Eigen::MatrixXf xi = cuben::ode::rk4Sys(system_function, ti, x0);
                Eigen::MatrixXf expected(3,2); expected <<
                    1.0f, 0.0f,
                    0.541667f, -0.833333f,
                    -0.401042f, -0.902778f;
                ASSERT_TRUE(cuben::fundamentals::isMatrixWithinReltol(xi, expected, 1e-3, true));
            }

            TEST(TestOde, Bs23SysTest) {
                Eigen::VectorXf x0(2); x0 <<
                    40.0f, 9.0f;
                Eigen::Vector2f tInt(0.0f, 20.0f);
                Eigen::VectorXf ti;
                Eigen::MatrixXf xi;
                cuben::ode::bs23Sys(predatorPrey, tInt, x0, ti, xi);

                // evaluate against t=10 and t=20, using CubicSpline
                cuben::CubicSpline xi0;
                cuben::CubicSpline xi1;
                const int I = ti.rows();
                for (int i = 0; i < I; i += 1) {
                    xi0.push(ti(i), xi(i,0));
                    xi1.push(ti(i), xi(i,1));
                }
                std::cout << "x(t=10) = [" << xi0.eval(10) << "," << xi1.eval(10) << "]" << std::endl;
                std::cout << "x(t=20) = [" << xi0.eval(20) << "," << xi1.eval(20) << "]" << std::endl;
                Eigen::VectorXf x10_act(2); x10_act <<
                    xi0.eval(10), xi1.eval(10);
                Eigen::VectorXf x20_act(2); x20_act <<
                    xi0.eval(20), xi1.eval(20);
                Eigen::VectorXf x10_ref(2); x10_ref <<
                    2.89684, 17.849;
                Eigen::VectorXf x20_ref(2); x20_ref <<
                    0.697467, 7.48224;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(x10_act, x10_ref, 1e-3, true));
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(x20_act, x20_ref, 1e-3, true));
            }

            TEST(TestOde, Dp45SysTest) {
                Eigen::VectorXf x0(2);
                x0(0) = 1.0f;
                x0(1) = 0.0f;
                Eigen::Vector2f tInt(0.0f, 10.0f);
                Eigen::VectorXf ti;
                Eigen::MatrixXf xi;
                cuben::ode::dp45Sys(vanDerPol, tInt, x0, ti, xi);

                // evaluate against t=5 and t=10, using cubic splines
                cuben::CubicSpline cs0;
                cuben::CubicSpline cs1;
                const int I = ti.rows();
                for (int i = 0; i < I; i += 1) {
                    cs0.push(ti(i), xi(i,0));
                    cs1.push(ti(i), xi(i,1));
                }
                std::cout << "x(t=5) = [" << cs0.eval(5) << "," << cs1.eval(5) << "]" << std::endl;
                std::cout << "x(t=10) = [" << cs0.eval(10) << "," << cs1.eval(10) << "]" << std::endl;
                Eigen::VectorXf x5_act(2); x5_act <<
                    cs0.eval(5), cs1.eval(5);
                Eigen::VectorXf x10_act(2); x10_act <<
                    cs0.eval(10), cs1.eval(10);
                Eigen::VectorXf x5_ref(2); x5_ref <<
                    0.196475f, 2.99799f;
                Eigen::VectorXf x10_ref(2); x10_ref <<
                    -1.94682f, 0.300796f;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(x5_act, x5_ref, 1e-3, true));
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(x10_act, x10_ref, 1e-3, true));
            }

            TEST(TestOde, ImpEulerTest) {
                const int N = 10;
                Eigen::VectorXf ti(N);
                for (int i = 0; i < N; i += 1) {
                    ti(i) = (float)i / 10.0f;
                }
                float x0 = 1.0f;
                Eigen::VectorXf xi = cuben::ode::impEuler(fInd, fImp, ti, x0);
                Eigen::VectorXf xi_ref(N); xi_ref <<
                    1.0f, 0.909091f, 0.826446f, 0.751315f, 0.683013f, 0.620921f, 0.564474f, 0.513158f, 0.466507f, 0.424098f;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(xi, xi_ref, 1e-3, true));
            }

            TEST(TestOde, ImpTrapTest) {
                const int N = 10;
                Eigen::VectorXf ti(N);
                for (int i = 0; i < N; i += 1) {
                    ti(i) = (float)i / 10.0f;
                }
                float x0 = 1.0f;
                Eigen::VectorXf xi = cuben::ode::impTrap(fInd, fImp, ti, x0);
                Eigen::VectorXf xi_ref(N); xi_ref <<
                    1.0f, 0.904762f, 0.818594f, 0.740633f, 0.670096f, 0.606278f, 0.548537f, 0.496295f, 0.449029f, 0.406264f;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(xi, xi_ref, 1e-3, true));
            }

            TEST(TestOde, ModAb2sTest) {
                const int N = 10;
                Eigen::VectorXf ti(N);
                for (int i = 0; i < N; i += 1) {
                    ti(i) = (float)i / 10.0f;
                }
                float x0 = 1.0f;
                Eigen::VectorXf xi = cuben::ode::modab2s(fInd, ti, x0);
                Eigen::VectorXf xi_ref(N); xi_ref << 
                    1.0f, 0.905f, 0.81925f, 0.741612f, 0.671333f, 0.607714f, 0.550123f, 0.49799f, 0.450798f, 0.408078f;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(xi, xi_ref, 1e-3, true));
            }

            TEST(TestOde, ModAb3sTest) {
                Eigen::VectorXf ti(5); ti <<
                    0.0, 1.0, 2.0, 3.0, 4.0;
                Eigen::VectorXf xi = cuben::ode::modab3s(another_test_dxdt, ti, 1.0f);
                Eigen::VectorXf expected(5); expected <<
                    1.0f, 0.905f, 0.862125f, 0.775884f, 0.704415f;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(xi, expected, 1e-3, true));
            }

            TEST(TestOde, ModAb4sTest) {
                Eigen::VectorXf ti(6); ti <<
                    0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f;
                Eigen::VectorXf xi = cuben::ode::modab4s(simpleODE, ti, 1.0f);
                Eigen::VectorXf expected(6); expected <<
                    1.0f, 1.22f, 1.353f, 1.62965f, 2.01251f, 2.45935f;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(xi, expected, 1e-3, true));
            }

            TEST(TestOde, ModAm2sTest) {
                Eigen::VectorXf ti(5); ti <<
                    0.0, 0.1, 0.2, 0.3, 0.4;
                Eigen::VectorXf result = cuben::ode::modam2s(fIndTest, fImpTest, ti, 1.0);
                Eigen::VectorXf expected(5); expected <<
                      1.0f, 1.00503f, 1.02023f, 1.04607f, 1.08335;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(result, expected, 1e-3, true));
            }

            TEST(TestOde, ModMs2sTest) {
                const int N = 10;
                Eigen::VectorXf ti(N);
                for (int i = 0; i < N; i += 1) {
                    ti(i) = (float)i / 10.0;
                }
                Eigen::VectorXf result = cuben::ode::modms2s(yet_another_test, yet_another_test, ti, 1.0);
                Eigen::VectorXf expected(N); expected <<
                    1.0f, 0.951219f, 0.904838f, 0.860698f, 0.818732f, 0.778791f, 0.74082f, 0.704679f, 0.670322f, 0.637619f;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(result, expected, 1e-3, true));
            }

            TEST(TestOde, ModAm3sTest) {
                const int N = 10;
                Eigen::VectorXf ti(N);
                for (int i = 0; i < N; i += 1) {
                    ti(i) = std::powf((float)i, 1.1f) / 10.0;
                }
                Eigen::VectorXf result = cuben::ode::modam3s(fInd, fImp, ti, 1.0f);
                Eigen::VectorXf expected(N); expected << 
                    1.0f, 0.904762f, 0.810484f, 0.731072f, 0.656147f, 0.583277f, 0.51594f, 0.454712f, 0.399601f, 0.350331f;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(result, expected, 1e-3, true));
            }

            TEST(TestOde, ModAm4sTest) {
                const int N = 10;
                Eigen::VectorXf ti(N);
                for (int i = 0; i < N; i += 1) {
                    ti(i) = std::powf((float)i, 1.1f) / 10.0;
                }
                Eigen::VectorXf result = cuben::ode::modam4s(harmonicOscillator, harmonicOscillator, ti, 2.0f);
                Eigen::VectorXf expected(N); expected <<
                    2.0f, 1.07692, 0.550925, 0.306448, 0.144486, 0.0819095, 0.0365785, 0.0198439, 0.00807169, 0.00442807;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(result, expected, 1e-3, true));
            }
        }
    }
}

int main(int nArgs, char** vArgs) {
    ::testing::InitGoogleTest(&nArgs, vArgs);
    return RUN_ALL_TESTS();
}
