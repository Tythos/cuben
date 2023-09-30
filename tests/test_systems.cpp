/**
 * tests/test_systems.cpp
 */

#include <iostream>
#include "gtest/gtest.h"
#include "cuben.hpp"

Eigen::VectorXf testf(Eigen::VectorXf x) {
	Eigen::VectorXf f(2);
	float u = x(0); float v = x(1);
	f(0) = 6.0f * u * u * u + u * v - 3.0f * v * v * v - 4.0f;
	f(1) = u * u - 18.0f * u * v * v + 16.0f * v * v * v + 1.0f;
	return f;
}

Eigen::MatrixXf testdfdx(Eigen::VectorXf x) {
	Eigen::MatrixXf dfdx(2,2);
	float u = x(0); float v = x(1);
	dfdx(0,0) = 18.0f * u * u + v;
	dfdx(0,1) = u - 9.0f * v * v;
	dfdx(1,0) = 2.0f * u - 18.0f * v * v;
	dfdx(1,1) = -36.0f * u * v + 48.0f * v * v;
	return dfdx;
}

namespace cuben {
	namespace tests {
		namespace test_systems {
			TEST(TestSystems, GaussianElimination) {
				Eigen::MatrixXf A(3,3); A << 1, 2, -1,   2, 1, -2,   -3, 1, 1;
				Eigen::VectorXf y(3); y << 3, 3, -6;
				Eigen::VectorXf x = cuben::systems::gaussElim(A, y);
				std::cout << "Gaussian elimination: "; cuben::fundamentals::printVecTrans(x); std::cout << std::endl;
				Eigen::VectorXf expected(3); expected << 3.0, 1.0, 2.0;
				ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(x, expected));
			}

			TEST(TestSystems, LuFactorization) {
				Eigen::MatrixXf A(3,3); A << 1, 2, -1,   2, 1, -2,   -3, 1, 1;
				Eigen::MatrixXf L(3,3), U(3,3);
				cuben::systems::luFactor(A, L, U);
				std::cout << "L:" << L << std::endl;
				std::cout << "U:" << U << std::endl;
				// need matrix assertion; difference-dominant diagonals?
				// ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(x, expected));
			}

			TEST(TestSystems, LuSolving) {
				Eigen::MatrixXf A(3,3); A << 1, 2, -1,   2, 1, -2,   -3, 1, 1;
				Eigen::VectorXf y(3); y << 3, 3, -6;
				Eigen::VectorXf x = cuben::systems::luSolve(A, y);
				std::cout << "LU factorziation: "; cuben::fundamentals::printVecTrans(x); std::cout << std::endl;
				Eigen::VectorXf expected(3); expected << 3.0, 1.0, 2.0;
				ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(x, expected));
			}

			TEST(TestSystems, LuResiduals) {
				Eigen::MatrixXf A(3,3); A << 1, 2, -1,   2, 1, -2,   -3, 1, 1;
				Eigen::VectorXf y(3); y << 3, 3, -6;
				Eigen::VectorXf x = cuben::systems::luSolve(A, y);
				Eigen::VectorXf r = cuben::systems::residual(A, y, x);
				std::cout << "LU residuals: "; cuben::fundamentals::printVecTrans(r); std::cout << std::endl;
				Eigen::VectorXf expected(3); expected << 0.0, 0.0, 0.0;
				ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(r, expected));
			}

			TEST(TestSystems, PaluFactorization) {
				Eigen::MatrixXf A(3,3); A << 1, 2, -1,   2, 1, -2,   -3, 1, 1;
				Eigen::VectorXf y(3); y << 3, 3, -6;
				Eigen::VectorXf x = cuben::systems::paluSolve(A, y);
				std::cout << "PALU factorziation: "; cuben::fundamentals::printVecTrans(x); std::cout << std::endl;
				Eigen::VectorXf expected(3); expected << 3.0, 1.0, 2.0;
				ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(x, expected));
			}

			TEST(TestSystems, TestBroyden) {
				std::cout.precision(16);
				Eigen::VectorXf x0(2); x0 << 1.00f,-0.50f;
				Eigen::VectorXf x1(2); x0 << 0.90f,-0.35f;
				for (int i = 0; i < 16; i++) {
					// cuben::constants::iterLimit = i + 1;
					std::cout << "x (" << i + 1 << " iterations): " << cuben::systems::broydenTwo(&testf, x0, x1, Eigen::MatrixXf::Identity(2,2)).transpose() << std::endl;
					// need better assertion/case here?
				}
			}
		}
	}
}

int main(int nArgs, char** vArgs) {
	::testing::InitGoogleTest(&nArgs, vArgs);
	return RUN_ALL_TESTS();
}
