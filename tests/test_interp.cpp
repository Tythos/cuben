/**
 * tests/test_interp.cpp
 */

#include "cuben.hpp"
#include "gtest/gtest.h"

namespace cuben {
	namespace tests {
		namespace test_interp {
			float testFunction(float x) {
				return std::sin(5 * x) + x * x * x;
			}

			TEST(TestInterpolation, LagangeTest) {
				Eigen::VectorXf xi(5); xi << 2,3,5,7,11;
				Eigen::VectorXf yi(5); yi << 1,2,4,8,16;
				{
					int i = 0;
					float x = 1.5 + 2 * (i + 1);
					float fx = cuben::interp::lagrange(xi, yi, x);
					std::cout << "f(" << x << ") = " << fx << std::endl;
					ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(fx, 2.4082, 1e-3));
				} {
					int i = 1;
					float x = 1.5 + 2 * (i + 1);
					float fx = cuben::interp::lagrange(xi, yi, x);
					std::cout << "f(" << x << ") = " << fx << std::endl;
					ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(fx, 4.77799, 1e-3));
				} {
					int i = 2;
					float x = 1.5 + 2 * (i + 1);
					float fx = cuben::interp::lagrange(xi, yi, x);
					std::cout << "f(" << x << ") = " << fx << std::endl;
					ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(fx, 9.31445, 1e-3));
				}
			}

			TEST(TestInterpolation, SinInterpTest) {
				for (float x = 0.0f; x <= 2 * M_PI; x += M_PI / 16) {
					float expected = std::sin(x);
					float result = cuben::interp::sinInterp(x);
					// std::cout << "sin(" << x << ") ~> " << result << std::endl;
					// ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(result, expected, 1e-3));
				}
			}

			TEST(TestInterpolation, ChebyshevTest) {
				float xMin = -1.0f;
				float xMax = 1.0f;
				int n = 10;
				cuben::Polynomial p = cuben::interp::chebyshev(testFunction, xMin, xMax, n);
				const float dx = (xMax - xMin) / 100;
				for (float x = xMin; x <= xMax; x += dx) {
					float expected = testFunction(x);
					float result = p.eval(x);
					// std::cout << "f(" << x << ") ~> " << result << std::endl;
					// ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(result, expected, 1e-3));
				}
			}

			TEST(TestInterpolation, ChebyPolyTest) {
				const int nSamples = 10;
				const int nMaxOrder = 3;
				Eigen::VectorXf theta(nSamples);
				for (int i = 0; i < nSamples; i++) {
					theta(i) = 2 * M_PI * i / (float)nSamples;
				}
				Eigen::VectorXf t = theta.array().cos();
				for (int order = 0; order <= nMaxOrder; order++) {
					Eigen::VectorXf result = cuben::interp::cheb(t, order);
					Eigen::VectorXf expected = theta.array() * order;
					expected = expected.array().cos();
					for (int i = 0; i < nSamples; i++) {
						ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(result(i), expected(i), 1e-3));
					}
				}
			}

			TEST(TestInterpolation, DchebyTest) {
				const int nSamples = 10;
				const int nMaxOrder = 3;
				Eigen::VectorXf t(nSamples);
				for (int i = 0; i < nSamples; i++) {
					t(i) = -1.0 + 2.0 * i / 99.0;
				}
				float h = 1e-6;
				for (int order = 0; order <= nMaxOrder; order++) {
					Eigen::VectorXf result = cuben::interp::dchebdt(t, order);
					Eigen::VectorXf expected(nSamples);
					for (int i = 0; i < nSamples; i++) {
						Eigen::VectorXf tPlus = Eigen::VectorXf::Constant(nSamples, t(i) + h);
						Eigen::VectorXf tMinus = Eigen::VectorXf::Constant(nSamples, t(i) - h);
						float fPlus = cuben::interp::cheb(tPlus, order)(i);
						float fMinus = cuben::interp::cheb(tMinus, order)(i);
						expected(i) = (fPlus - fMinus) / (2.0 * h);
					}
					for (int i = 0; i < nSamples; i++) {
						// std::cout << "result: " << result(i) << "; expected: " << expected(i) << std::endl;
						// ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(result(i), expected(i), 1e-3));
					}
				}
			}

			TEST(TestInterpolation, D2chebyTest) {
				const int nSamples = 100;
				const int nMaxOrder = 3;
				Eigen::VectorXf t(nSamples);  // Sample t-values between -1 and 1
				for (int i = 0; i < nSamples; i++) {
					t(i) = -1.0 + 2.0 * i / (float)(nSamples - 1);
				}
				float h = 1e-6;
				for (int order = 0; order <= nMaxOrder; order++) {
					Eigen::VectorXf result = cuben::interp::d2chebdt2(t, order);
					Eigen::VectorXf expected(nSamples);
					for (int i = 0; i < nSamples; i++) {
						Eigen::VectorXf tPlus = Eigen::VectorXf::Constant(nSamples, t(i) + h);
						Eigen::VectorXf tMinus = Eigen::VectorXf::Constant(nSamples, t(i) - h);
						float fPlus = cuben::interp::cheb(tPlus, order)(i);
						float fMinus = cuben::interp::cheb(tMinus, order)(i);
						float fCenter = cuben::interp::cheb(t, order)(i);
						expected(i) = (fPlus - 2.0 * fCenter + fMinus) / (h * h);
					}
					for (int i = 0; i < nSamples; i++) {
						// std::cout << "result: " << result(i) << "; expected: " << expected(i) << std::endl;
						// ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(result(i), expected(i), 1e-3));
					}
				}
			}

			TEST(TestInterpolation, ChebySampTest) {
				float lhs = -1.0;
				float rhs = 1.0;
				int n = 7;
				Eigen::VectorXf xi = cuben::interp::chebSamp(lhs, n, rhs);
				cuben::fundamentals::printVecTrans(xi);
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(xi(0), -1.0, 1e-3));
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(xi(1), -8.6603e-1, 1e-3));
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(xi(2), -5.0000e-1, 1e-3));
				EXPECT_NEAR(xi(3), 0.0, 1e-3);
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(xi(4), 5.0000e-1, 1e-3));
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(xi(5), 8.6603e-1, 1e-3));
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(xi(6), 1.0, 1e-3));
			}
		}
	}
}

int main(int nArgs, char** vArgs) {
	::testing::InitGoogleTest(&nArgs, vArgs);
	return RUN_ALL_TESTS();
}
