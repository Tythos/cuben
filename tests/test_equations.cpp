/**
 * tests/test_equations.cpp
 */

#include <iostream>
#include "gtest/gtest.h"
#include "cuben.hpp"

float f(float x) {
	return x * x * x + x - 1;
}

float f0(float x) {
	return (1 + 2 * x * x * x) / (1 + 3 * x * x);
}

float dfdx(float x) {
	return 3 * x * x + 1;
}

namespace cuben {
	namespace tests {
		namespace test_equations {
			TEST(TestEquations, BisectionTest) {
				// test bisection
				float y = cuben::equations::bisect(f, 0., 1.);
				std::cout << "Bisection solution: " << y << std::endl;
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(y, 6.82328e-1));
			}

			TEST(TestEquations, FpiTest) {
				// test floating-point iteration
				float y = cuben::equations::fpi(f0, 0.5);
				std::cout << "Fixed point iteration: " << y << std::endl;
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(y, 6.82328e-1));
			}

			TEST(TestEquations, NewtTest) {
				// test newton's method
				float y = cuben::equations::newt(f, dfdx, -0.7);
				std::cout << "Newton solution: " << y << std::endl;
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(y, 6.82328e-1));
			}

			TEST(TestEquations, ModNewtTest) {
				// test modified newton's method
				float y = cuben::equations::modNewt(f, dfdx, 1., 1.);
				std::cout << "Modified Newton solution: " << y << std::endl;
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(y, 6.82328e-1));
			}

			TEST(TestEquations, SecantTest) {
				// test secant
				float y = cuben::equations::secant(f, 0.0, 1.0);
				std::cout << "Secant solution: " << y << std::endl;
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(y, 6.82328e-1));
			}

			TEST(TestEquations, RegulaFalsiTest) {
				// test regula falsi
				float y = cuben::equations::regulaFalsi(f, 0., 1.);
				std::cout << "Regula falsi solution: " << y << std::endl;
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(y, 6.82328e-1));
			}

			TEST(TestEquations, MullerTest) {
				// test muller's method
				float y = cuben::equations::muller(f, 0., 1.);
				std::cout << "Muller solution: " << y << std::endl;
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(y, 6.82328e-1));
			}

			TEST(TestEquation, IqiTest) {
				// test iqi
				float y = cuben::equations::iqi(f, 0., 1.);
				std::cout << "Inverse quadratic interpolation: " << y << std::endl;
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(y, 6.82328e-1));
			}

			TEST(TestEquations, BrentsTest) {
				// test brent's method
				float y = cuben::equations::brents(f, 0, 1);
				std::cout << "Brent's method: " << y << std::endl;
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(y, 6.82328e-1));
			}
		}
	}
}

int main(int nArgs, char** vArgs) {
	::testing::InitGoogleTest(&nArgs, vArgs);
	return RUN_ALL_TESTS();
}
