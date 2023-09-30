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

Eigen::VectorXf myFunction(Eigen::VectorXf x) {
	Eigen::VectorXf result(2);
	result(0) = x(0) * x(0) + x(1) * x(1) - 2;
	result(1) = x(0) * x(0) - x(1) * x(1);
	return result;
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
				Eigen::MatrixXf A(2,2); A << 2, 1, 1, 3;
				Eigen::MatrixXf L(2,2), U(2,2);
				cuben::systems::luFactor(A, L, U);
				std::cout << "L:" << L << std::endl;
				std::cout << "U:" << U << std::endl;
				Eigen::MatrixXf expectedL(2,2); expectedL << 1, 0, 0.5, 1;
				Eigen::MatrixXf expectedU(2,2); expectedU << 2, 1, 0, 2.5;
				ASSERT_TRUE(cuben::fundamentals::isMatrixWithinReltol(L, expectedL));
				ASSERT_TRUE(cuben::fundamentals::isMatrixWithinReltol(U, expectedU));
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
				Eigen::VectorXf x0(2); x0 << 0.5, 0.5;
				Eigen::VectorXf x1(2); x1 << 0.6, 0.6;
				Eigen::MatrixXf B0 = Eigen::MatrixXf::Identity(2,2);
				Eigen::VectorXf solution = cuben::systems::broydenTwo(myFunction, x0, x1, B0);
				std::cout << "Broyden's method: "; cuben::fundamentals::printVecTrans(solution); std::cout << std::endl;
				Eigen::VectorXf expected(2); expected << 1.0, 1.0;
				ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(solution, expected));
			}
		}
	}
}

int main(int nArgs, char** vArgs) {
	::testing::InitGoogleTest(&nArgs, vArgs);
	return RUN_ALL_TESTS();
}
