/**
 * tests/test_leastsq.cpp
 */

#include "cuben.hpp"
#include "gtest/gtest.h"

namespace cuben {
    namespace tests {
        namespace test_leastsq {
            Eigen::VectorXf fSample(Eigen::VectorXf x) {
                Eigen::VectorXf f(4);
                f(0) = std::sqrt((x(0) + 1) * (x(0) + 1) + x(1) * x(1)) - (1 + x(2));
                f(1) = std::sqrt((x(0) - 1) * (x(0) - 1) + (x(1) - 0.5) * (x(1) - 0.5)) - (0.5 + x(2));
                f(2) = std::sqrt((x(0) - 1) * (x(0) - 1) + (x(1) + 0.5) * (x(1) + 0.5)) - (0.5 + x(2));
                f(3) = std::sqrt(x(0) * x(0) + (x(1) - 1) * (x(1) - 1)) - (0.5 + x(2));
                return f;
            }

            Eigen::MatrixXf dfdxSample(Eigen::VectorXf x) {
                Eigen::MatrixXf dfdx(4,3);
                Eigen::VectorXf f = fSample(x);
                dfdx(0,0) = (x(0) + 1) / (f(0) + 1);
                dfdx(0,1) = x(1) / (f(0) + 1);
                dfdx(0,2) = -1;
                dfdx(1,0) = (x(0) - 1) / (f(1) + 0.5);
                dfdx(1,1) = (x(1) - 0.5) / (f(1) + 0.5);
                dfdx(1,2) = -1;
                dfdx(2,0) = (x(0) - 1) / (f(2) + 0.5);
                dfdx(2,1) = (x(1) + 0.5) / (f(2) + 0.5);
                dfdx(2,2) = -1;
                dfdx(3,0) = x(0) / (f(3) + 0.5);
                dfdx(3,1) = (x(1) - 1) / (f(3) + 0.5);
                dfdx(3,2) = -1;
                return dfdx;
            }

            TEST(TestLeastSq, OriginalTest) {
                Eigen::VectorXf x = cuben::leastsq::nonLinearGaussNewton(fSample, dfdxSample, Eigen::VectorXf::Zero(3), 5);
                std::cout << "x:" << std::endl << x << std::endl << std::endl;
            }

            TEST(TestLeastSq, InvertNormalTest) {
                Eigen::MatrixXf A(3, 3);
                A << 2, 0, 0,
                    0, 3, 0,
                    0, 0, 4;
                Eigen::VectorXf y(3);
                y << 2, 6, 16;
                Eigen::VectorXf x;
                x = cuben::leastsq::invertNormal(A, y);
                Eigen::VectorXf y_predicted = A * x;
                for (int i = 0; i < y.rows(); i += 1) {
                    EXPECT_NEAR(y[i], y_predicted[i], 1e-5);
                }
            }

			TEST(TestLeastSq, InvertSvdSimple) {
                Eigen::MatrixXf A(3, 3);
                A << 2, 0, 0,
                    0, 3, 0,
                    0, 0, 4;
                Eigen::VectorXf y(3);
                y << 2, 6, 16;
                Eigen::VectorXf x;
                x = cuben::leastsq::invertSvd(A, y);
                Eigen::VectorXf y_predicted = A * x;
                for (int i = 0; i < y.rows(); i += 1) {
                    EXPECT_NEAR(y[i], y_predicted[i], 1e-5);
                }
			}

			TEST(TestLeastSq, InvertSvdIll) {
				Eigen::MatrixXf A(3,3);
				A << 1, 1, 1.001,
					1, 1.001, 1,
					1.001, 1, 1;
				Eigen::VectorXf y(3);
				y << 1, 2, 3;
				Eigen::VectorXf result = cuben::leastsq::invertSvd(A, y);
				cuben::fundamentals::printVecTrans(result);
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(result(0), 1.00063e3));
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(result(1), 6.59241e-1));
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(result(2), -9.99289e2));
			}

            TEST(TestLeastSq, InvertNormalFail) {
                Eigen::MatrixXf A(3, 3);
                A << 1, 2, 3,
                    4, 5, 6,
                    7, 8, 9;
                Eigen::VectorXf y(2);
                y << 1, 2;
                EXPECT_THROW(cuben::leastsq::invertNormal(A, y), cuben::exceptions::xMismatchedDims);                
            }

            TEST(TestLeastSq, FitPolyTest) {
                Eigen::VectorXf xi(2);
                Eigen::VectorXf yi(2);
                xi << 1, 2;
                yi << 2, 4;
                Eigen::VectorXf coeffs = cuben::leastsq::fitPolynomial(xi, yi, 1);
                EXPECT_NEAR(coeffs[0], 0.0, 1e-3);
                EXPECT_NEAR(coeffs[1], 2.0, 1e-3);
            }

            TEST(TestLeastSq, FitPolyQuadTest) {
                Eigen::VectorXf xi(3);
                Eigen::VectorXf yi(3);
                xi << 1, 2, 3;
                yi << 1, 4, 9;
                Eigen::VectorXf coeffs = cuben::leastsq::fitPolynomial(xi, yi, 2);
                EXPECT_NEAR(coeffs[0], 0.0, 1e-3);
                EXPECT_NEAR(coeffs[1], 0.0, 1e-3);
                EXPECT_NEAR(coeffs[2], 1.0, 1e-3);                
            }

            TEST(TestLeastSq, FitPolyFail) {
                Eigen::VectorXf xi(3);
                Eigen::VectorXf yi(2);
                xi << 1, 2, 3;
                yi << 1, 2;
                EXPECT_THROW(cuben::leastsq::fitPolynomial(xi, yi, 1), cuben::exceptions::xMismatchedDims);
            }

			TEST(TestLeastSq, FitPeriodicSimple) {
				Eigen::VectorXf xi(3);
				xi << 0, 0.5, 1.0;
				Eigen::VectorXf yi(3);
				yi << 1, -1, 1;
				Eigen::VectorXf result = cuben::leastsq::fitPeriodic(xi, yi, 1);
				EXPECT_NEAR(0, result[0], 1e-5);
				EXPECT_NEAR(1, result[1], 1e-5);
				EXPECT_NEAR(0, result[2], 1e-5);
			}

			TEST(TestLeastSq, FitPeriodicFail) {
				Eigen::VectorXf xi(3);
				Eigen::VectorXf yi(2);
				xi << 0, M_PI/2, M_PI;
				yi << 0, 1;
				EXPECT_THROW(cuben::leastsq::fitPeriodic(xi, yi, 1), cuben::exceptions::xMismatchedDims);				
			}

			TEST(TestLeastSq, FitPeriodicConstant) {
				Eigen::VectorXf xi(3);
				Eigen::VectorXf yi(3);
				xi << 0, M_PI/2, M_PI;
				yi << 1, 1, 1;
				Eigen::VectorXf coeffs = cuben::leastsq::fitPeriodic(xi, yi, 1);
				EXPECT_NEAR(coeffs[0], 1.0, 1e-5);
				EXPECT_NEAR(coeffs[1], 0.0, 1e-5);
				EXPECT_NEAR(coeffs[2], 0.0, 1e-5);
			}

			TEST(TestLeastSq, FitExpoBasic) {
				float a_true = 2.0;
				float b_true = 1.5;
				int n = 100;
				Eigen::VectorXf xi(n);
				Eigen::VectorXf yi(n);
				float noiseStd = 0.05;
				for (int i = 0; i < n; i++) {
					xi(i) = i * 0.1;
					yi(i) = a_true * exp(b_true * xi(i)) + noiseStd * ((rand() % 1000) / 1000.0 - 0.5);
				}
				Eigen::VectorXf c = cuben::leastsq::fitExponential(xi, yi);
				float a_est = c(0);
				float b_est = c(1);
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(a_est, a_true, 1e-3));
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(b_est, b_true, 1e-3));
			}

			TEST(TestLeastSq, FitPowerBasic) {
				Eigen::VectorXf xi(3);
				Eigen::VectorXf yi(3);
				xi << 1, 2, 3;
				yi << 2, 16, 54;
				Eigen::VectorXf coefficients = cuben::leastsq::fitPower(xi, yi);
	   			ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(coefficients(0), 2, 1e-3));
    			ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(coefficients(1), 3, 1e-3));
			}

			TEST(TestLeastSq, FitGammaBasic) {
				Eigen::VectorXf xi(3), yi(3);
				xi << 1, 2, 3;
				yi << 1, 4, 9;
				Eigen::VectorXf coeffs = cuben::leastsq::fitGamma(xi, yi);
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(coeffs(0), 1.0, 1e-3, true)); 
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(coeffs(1), 2.0, 1e-3, true)); 
			}

			TEST(TestLeastSq, FitCompoundBasic) {
				const float E = std::exp(1);
				Eigen::VectorXf xi(3);
				Eigen::VectorXf yi(3);
				xi << 1, 2, 3;
				yi << E, E * E, E * E * E;
				Eigen::VectorXf coeffs = cuben::leastsq::fitCompoundExpo(xi, yi);
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(coeffs(0), 1.0, 1e-3, true));
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(coeffs(1), 0.0, 1e-3, true));
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(coeffs(2), 1.0, 1e-3, true));				
			}

			TEST(TestLeastSq, GramSchmitBasic) {
				Eigen::MatrixXf A(3, 3);
				A << 1, 2, 3,
					0, 1, 4,
					0, 0, 1;
				Eigen::MatrixXf Q = cuben::leastsq::computeGramSchmidt(A);
				for (int i = 0; i < 3; i++) {
					ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(Q.col(i).norm(), 1.0f, 1e-6));
				}
				for (int i = 0; i < 2; i++) {
					for (int j = i+1; j < 3; j++) {
						ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(Q.col(i).dot(Q.col(j)), 0.0f, 1e-6, true));
					}
				}
			}

			TEST(TestLeastSq, QrBasic) {
				Eigen::MatrixXf A(3, 3);
				A << 1, 2, 3,
					0, 1, 4,
					0, 0, 1;
				Eigen::MatrixXf Q, R;
				cuben::leastsq::qrFactor(A, Q, R);
				Eigen::MatrixXf I = Q.transpose() * Q;
				for (int i = 0; i < 3; i++) {
					for (int j = 0; j < 3; j++) {
						float expected = (i == j) ? 1.0f : 0.0f;
						ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(I(i,j), expected, 1e-6, true));
					}
				}
				Eigen::MatrixXf computedA = Q * R;
				for (int i = 0; i < 3; i++) {
					for (int j = 0; j < 3; j++) {
						ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(computedA(i,j), A(i,j), 1e-6, true));
					}
				}
			}
        }
    }
}

int main(int nArgs, char** vArgs) {
    ::testing::InitGoogleTest(&nArgs, vArgs);
    return RUN_ALL_TESTS();
}
