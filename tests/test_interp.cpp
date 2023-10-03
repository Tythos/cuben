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
		}
	}
}

int main(int nArgs, char** vArgs) {
	::testing::InitGoogleTest(&nArgs, vArgs);
	return RUN_ALL_TESTS();
}
