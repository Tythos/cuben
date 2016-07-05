#include <iostream>
#include "cuben.h"

int main(int nArgs, char** vArgs) {
	Eigen::MatrixXf A(3,3); A << 1, 2, -1,   2, 1, -2,   -3, 1, 1;
	Eigen::VectorXf y(3); y << 3, 3, -6;
	Eigen::VectorXf x = Cuben::Systems::gaussElim(A, y);
	std::cout << "Gaussian elimination: "; Cuben::Fund::printVecTrans(x); std::cout << std::endl;
	Eigen::MatrixXf L(3,3), U(3,3);
	Cuben::Systems::luFactor(A, L, U);
	std::cout << "L:" << L << std::endl;
	std::cout << "U:" << U << std::endl;
	x = Cuben::Systems::luSolve(A, y);
	std::cout << "LU factorziation: "; Cuben::Fund::printVecTrans(x); std::cout << std::endl;
	Eigen::VectorXf r = Cuben::Systems::residual(A, y, x);
	std::cout << "LU residuals: "; Cuben::Fund::printVecTrans(r); std::cout << std::endl;
	x = Cuben::Systems::paluSolve(A, y);
	std::cout << "PALU factorziation: "; Cuben::Fund::printVecTrans(x); std::cout << std::endl;
	Cuben::Systems::test();
	return 0;
}
