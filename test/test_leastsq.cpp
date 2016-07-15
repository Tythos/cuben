#include <iostream>
#include "cuben.h"

Eigen::VectorXf nonLinearGaussNewton(Eigen::VectorXf(*f)(Eigen::VectorXf), Eigen::MatrixXf(*dfdx)(Eigen::VectorXf), Eigen::VectorXf x0, int nIter) {
	Eigen::VectorXf f0 = f(x0);
	Eigen::MatrixXf dfdx0 = dfdx(x0);
	Eigen::MatrixXf A;
	Eigen::VectorXf b;
	Eigen::VectorXf dx(x0.rows()); dx(0) = Cuben::iterTol + 1.0f;
	int nf = f0.rows();
	int nx = x0.rows();
	if (nf != dfdx0.rows() || dfdx0.cols() != nx) {
		throw Cuben::xMismatchedDims();
	}
	
	int n = 0;
	while (dx.sum() > Cuben::iterTol && n < Cuben::iterLimit) {
//			for (int i = 0; i < nIter; i++) {
		n++;
		A = dfdx0.transpose() * dfdx0;
		b = -dfdx0.transpose() * f0;
		dx = A.inverse() * b;
		x0 = x0 + dx;
		f0 = f(x0);
		dfdx0 = dfdx(x0);
	}
	return x0;
}

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

int main(int nArgs, char** vArgs) {
	Eigen::VectorXf x(11); x << 2,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8,4.0;
	Eigen::VectorXf y(11);
	Eigen::MatrixXf A(11,8);
	for (int i = 0; i < 11; i++) {
		y(i) = 1  +  x(i)  +  x(i) * x(i)  +  x(i) * x(i) * x(i)  +  x(i) * x(i) * x(i) * x(i)  +  x(i) * x(i) * x(i) * x(i) * x(i) + x(i) * x(i) * x(i) * x(i) * x(i) * x(i) + x(i) * x(i) * x(i) * x(i) * x(i) * x(i) * x(i);
		A(i,0) = 1;
		for (int j = 1; j < 8; j++) {
			A(i,j) = A(i,j-1) * x(i);
		}
	}
	std::cout << "A: " << std::endl << A << std::endl << std::endl;
	std::cout << "y: " << std::endl << y << std::endl << std::endl;
	std::cout << "x: " << std::endl << Cuben::LeastSq::qrLeastSq(A, y) << std::endl << std::endl;
	x = nonLinearGaussNewton(fSample, dfdxSample, Eigen::VectorXf::Zero(3), 5);
	std::cout << "x:" << std::endl << x << std::endl << std::endl;
	return 0;
}
