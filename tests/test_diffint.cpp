/**
 * tests/test_diffint.cpp
 */

#include "cuben.hpp"
#include "gtest/gtest.h"

float f(float x) {
	return std::exp(-0.5f * x * x);
}

float f1(float x) {
	return 1.0f / x;
}

float f2(float x) {
	return log(x);
}

float fe(float x) {
	return exp(x);
}

namespace cuben {
	namespace tests {
		namespace test_diffint {
			TEST(TestDiffint, OldGaussQuad) {
				{
					int i = 2;
					float fx = cuben::diffint::intfdx_gaussQuad(f, -1.0f, 1.0f, i);
					std::cout << "int{f dx} [degree " << i << "] = " << fx << std::endl;
					ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(fx, 1.69296, 1e-3, true));
				} {
					int i = 3;
					float fx = cuben::diffint::intfdx_gaussQuad(f, -1.0f, 1.0f, i);
					std::cout << "int{f dx} [degree " << i << "] = " << fx << std::endl;
					ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(fx, 1.71202, 1e-3, true));
				} {
					int i = 4;
					float fx = cuben::diffint::intfdx_gaussQuad(f, -1.0f, 1.0f, i);
					std::cout << "int{f dx} [degree " << i << "] = " << fx << std::endl;
					ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(fx, 1.71122, 1e-3, true));
				}
			}

			TEST(TestDiffint, OldDiffTests) {
				float x = 2.0;
				float h = 0.1;
				{
					float dfdx = cuben::diffint::dfdx_2pfd(f1, x, h);
					std::cout << "2-point forward difference of f(x) = 1/x @ x = " << x << " w/ h = " << h << ":" << std::endl;
					std::cout << "    " << dfdx << std::endl;
					ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(dfdx, -0.238095, 1e-3, true));
				} {
					float dfdx = cuben::diffint::dfdx_3pcd(f1, x, h);
					std::cout << "3-point centered difference of f(x) = 1/x @ x = " << x << " w/ h = " << h << ":" << std::endl;
					std::cout << "    " << dfdx << std::endl;
					ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(dfdx, -0.250627, 1e-3, true));
				}
				std::cout << "h\t2pfd\t3pcd" << std::endl;
				{
					int i = 0;
					float h = pow(10,-i);
					float dfdx2p = cuben::diffint::dfdx_2pfd(fe, 0.0, h);
					float dfdx3p = cuben::diffint::dfdx_3pcd(fe, 0.0, h);
					std::cout << h << "\t" << dfdx2p << "\t" << dfdx3p << std::endl;
					ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(dfdx2p, 1.71828, 1e-3, true));
					ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(dfdx3p, 1.1752, 1e-3, true));
				} {
					int i = 1;
					float h = pow(10,-i);
					float dfdx2p = cuben::diffint::dfdx_2pfd(fe, 0.0, h);
					float dfdx3p = cuben::diffint::dfdx_3pcd(fe, 0.0, h);
					std::cout << h << "\t" << dfdx2p << "\t" << dfdx3p << std::endl;
					ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(dfdx2p, 1.05171, 1e-3, true));
					ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(dfdx3p, 1.00167, 1e-3, true));
				} {
					int i = 2;
					float h = pow(10,-i);
					float dfdx2p = cuben::diffint::dfdx_2pfd(fe, 0.0, h);
					float dfdx3p = cuben::diffint::dfdx_3pcd(fe, 0.0, h);
					std::cout << h << "\t" << dfdx2p << "\t" << dfdx3p << std::endl;
					ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(dfdx2p, 1.00502, 1e-3, true));
					ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(dfdx3p, 1.00002, 1e-3, true));
				} {
					int i = 3;
					float h = pow(10,-i);
					float dfdx2p = cuben::diffint::dfdx_2pfd(fe, 0.0, h);
					float dfdx3p = cuben::diffint::dfdx_3pcd(fe, 0.0, h);
					std::cout << h << "\t" << dfdx2p << "\t" << dfdx3p << std::endl;
					ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(dfdx2p, 1.00052, 1e-3, true));
					ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(dfdx3p, 1.00002, 1e-3, true));
				} {
					int i = 4;
					float h = pow(10,-i);
					float dfdx2p = cuben::diffint::dfdx_2pfd(fe, 0.0, h);
					float dfdx3p = cuben::diffint::dfdx_3pcd(fe, 0.0, h);
					std::cout << h << "\t" << dfdx2p << "\t" << dfdx3p << std::endl;
					ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(dfdx2p, 1.00017, 1e-3, true));
					ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(dfdx3p, 1.00017, 1e-3, true));
				} {
					int i = 5;
					float h = pow(10,-i);
					float dfdx2p = cuben::diffint::dfdx_2pfd(fe, 0.0, h);
					float dfdx3p = cuben::diffint::dfdx_3pcd(fe, 0.0, h);
					std::cout << h << "\t" << dfdx2p << "\t" << dfdx3p << std::endl;
					ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(dfdx2p, 1.00136, 1e-3, true));
					ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(dfdx3p, 1.00136, 1e-3, true));
				} {
					int i = 6;
					float h = pow(10,-i);
					float dfdx2p = cuben::diffint::dfdx_2pfd(fe, 0.0, h);
					float dfdx3p = cuben::diffint::dfdx_3pcd(fe, 0.0, h);
					std::cout << h << "\t" << dfdx2p << "\t" << dfdx3p << std::endl;
					ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(dfdx2p, 0.953674, 1e-3, true));
					ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(dfdx3p, 0.983477, 1e-3, true));
				} {
					int i = 7;
					float h = pow(10,-i);
					float dfdx2p = cuben::diffint::dfdx_2pfd(fe, 0.0, h);
					float dfdx3p = cuben::diffint::dfdx_3pcd(fe, 0.0, h);
					std::cout << h << "\t" << dfdx2p << "\t" << dfdx3p << std::endl;
					ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(dfdx2p, 1.19209, 1e-3, true));
					ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(dfdx3p, 1.19209, 1e-3, true));
				}
				h = 0.01;
				float d2fdx2 = cuben::diffint::d2fdx2_5pcd(f1, 2, 0.1);
				std::cout << "5-point centered second-difference of f(x) = 1/x @ x = " << x << " w/ h = " << h << ":" << std::endl;
				std::cout << "    " << d2fdx2 << std::endl;
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(d2fdx2, 0.249992, 1e-3, true));
			}

			TEST(TestDiffInt, OldIntTests) {
				float x0 = 1.0;
				float xF = 2.0;
				{
					float fdx = cuben::diffint::intfdx_trap(f2, x0, xF);
					std::cout << "Trapezoidally-integrated f(x) = ln(x), from " << x0 << " to " << xF << ":" << std::endl;
					std::cout << "    " << fdx << std::endl;
					ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(fdx, 0.346574, 1e-3, true));
				} {
					float fdx = cuben::diffint::intfdx_simp(f2, x0, xF);
					std::cout << "Simpson-integrated f(x) = ln(x), from " << x0 << " to " << xF << ":" << std::endl;
					std::cout << "    " << fdx << std::endl;
					ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(fdx, 0.385835, 1e-3, true));
				}
			}

			TEST(TestDiffInt, SecondOrder3pcdTest) {
				float x = 2.3;
				float h = 0.1;
				float d2fdx2 = cuben::diffint::d2fdx2_3pcd(f, x, h);
				std::cout << "Second-order 3-point central difference at x=" << x << " w/ h=" << h << std::endl;
				std::cout << "\td2fdx2 = " << d2fdx2 << std::endl;
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(d2fdx2, 0.304569, 1e-3, true));
			}

			TEST(TestDiffInt, FirstOrder5pcdTest) {
				float x = 2.3;
				float h = 0.1;
				float dfdx = cuben::diffint::dfdx_5pcd(f, x, h);
				std::cout << "First-order 5-point central difference at x=" << x << " w/ h=" << h << std::endl;
				std::cout << "\tdfdx = " << dfdx << std::endl;
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(dfdx, -0.163313, 1e-3, true));
			}

			TEST(TestDiffInt, DfdxCubicTest) {
				float x = 2.3;
				float h = 0.1;
				float dfdx = cuben::diffint::dfdx_cubic(f, x, h);
				std::cout << "Cubic difference at x=" << x << " w/ h=" << h << std::endl;
				std::cout << "\tdfdx = " << dfdx << std::endl;
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(dfdx, -0.163311, 1e-3, true));
			}

			TEST(TestDiffInt, IntSimp38Test) {
				float x0 = 1.0;
				float xF = 2.0;
				float fdx = cuben::diffint::intfdx_simp38(f, x0, xF);
				std::cout << "3-8 Simpsons integration between [" << x0 << "," << xF << "]" << std::endl;
				std::cout << "\tfdx = " << fdx << std::endl;
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(fdx, 0.340407, 1e-3, true));
			}

			TEST(TestDiffInt, IntMidTest) {
				float x0 = 1.0;
				float xF = 2.0;
				float fdx = cuben::diffint::intfdx_mid(f, x0, xF);
				std::cout << "Midpoint integration between [" << x0 << "," << xF << "]" << std::endl;
				std::cout << "\tfdx = " << fdx << std::endl;
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(fdx, 0.324652, 1e-3, true));
			}

			TEST(TestDiffInt, IntCompTrapTest) {
				float x0 = 1.0;
				float xF = 2.0;
				int n = 5;
				float fdx = cuben::diffint::intfdx_compTrap(f, x0, xF, n);
				std::cout << "Compound trapezoidal integration between [" << x0 << "," << xF << "] (n=" << n << ")" << std::endl;
				std::cout << "\tfdx = " << fdx << std::endl;
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(fdx, 0.341786, 1e-3, true));
			}

			TEST(TestDiffInt, IntCompSimpTest) {
				float x0 = 1.0;
				float xF = 2.0;
				int n = 5;
				float fdx = cuben::diffint::intfdx_compSimp(f, x0, xF, n);
				std::cout << "Compound Simpsons integration between [" << x0 << "," << xF << "] (n=" << n << ")" << std::endl;
				std::cout << "\tfdx = " << fdx << std::endl;
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(fdx, 0.340663, 1e-3, true));
			}

			TEST(TestDiffInt, IntCompMidTest) {
				float x0 = 1.0;
				float xF = 2.0;
				int n = 5;
				float fdx = cuben::diffint::intfdx_compMid(f, x0, xF, n);
				std::cout << "Compound midpoint integration between [" << x0 << "," << xF << "] (n=" << n << ")" << std::endl;
				std::cout << "\tfdx = " << fdx << std::endl;
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(fdx, 0.340101, 1e-3, true));
			}

			TEST(TestDiffInt, IntRombergTest) {
				float x0 = 1.0;
				float xF = 2.0;
				int n = 5;
				float fdx = cuben::diffint::intfdx_romberg(f, x0, xF, n);
				std::cout << "Romberg integration between [" << x0 << "," << xF << "] (n=" << n << ")" << std::endl;
				std::cout << "\tfdx = " << fdx << std::endl;
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(fdx, 0.340664, 1e-3, true));
			}

			TEST(TestDiffInt, IntAdaptTrapTest) {
				float x0 = 1.0;
				float xF = 2.0;
				float tol = 1e-3;
				float fdx = cuben::diffint::intfdx_adaptTrap(f, x0, xF, tol);
				std::cout << "Adaptive trapezoidal integration between [" << x0 << "," << xF << "] (tol=" << tol << ")" << std::endl;
				std::cout << "\tfdx = " << fdx << std::endl;
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(fdx, 0.341101, 1e-3, true));
			}

			TEST(TestDiffInt, IntAdaptSimpTest) {
				float x0 = 1.0;
				float xF = 2.0;
				float tol = 1e-3;
				float fdx = cuben::diffint::intfdx_adaptSimp(f, x0, xF, tol);
				std::cout << "Adaptive Simpsons integration between [" << x0 << "," << xF << "] (tol=" << tol << ")" << std::endl;
				std::cout << "\tfdx = " << fdx << std::endl;
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(fdx, 0.34063, 1e-3, true));
			}
		}
	}
}

int main(int nArgs, char** vArgs) {
	::testing::InitGoogleTest(&nArgs, vArgs);
	return RUN_ALL_TESTS();
}
