#include <iostream>
#include <cmath>
#include "cuben.h"

int main(int nArgs, char** vArgs) {
	// Polynomial tests
	Cuben::Fund::Polynomial p;
	p.push(0,2);
	p.push(0,3);
	p.push(0,-3);
	p.push(0,5);
	p.push(0,-1);
	std::cout << "Number of points: " << p.size() << std::endl;
	p.print();
	float x = 0.5;
	std::cout << "P(" << x << ") = " << p.eval(x) << std::endl;
	
	// Computational metrics tests
	float inf = std::numeric_limits<float>::infinity();
	float nan = std::nan("");
	float fin = 3.14;
	std::cout << inf << " is infinity? " << (Cuben::Fund::isInf(inf) == true ? "true" : "false") << std::endl;
	std::cout << nan << " is nan? " << (Cuben::Fund::isNan(nan) == true ? "true" : "false") << std::endl;
	std::cout << fin << " is fin? " << (Cuben::Fund::isFin(fin) == true ? "true" : "false") << std::endl;
	std::cout << "Machine epsilon: " << Cuben::Fund::machEps() << std::endl;
	std::cout << "Relative epsilon (to 10.): " << Cuben::Fund::relEps(10.) << std::endl;
	
	// Custom vector manipulations
	Eigen::VectorXf v(3);
	v << 1., 2., 4.;
	Cuben::Fund::printVecTrans(v);
	std::cout << std::endl << "sigma(v) = " << Cuben::Fund::stdDev(v) << std::endl;
	float val = 2.0f;
	std::cout << "Value " << val << " is in index " << Cuben::Fund::findValue(v, val) << std::endl;
	Eigen::Vector2i sub; sub << 2, 1;
	Eigen::Vector2i dims; dims << 3, 3;
	int ndx = Cuben::Fund::sub2ind(dims, sub);
	std::cout << "Subindex [" << sub(0) << "," << sub(1) << "], in a [" << dims(0) << " x " << dims(1) << "] matrix, is index " << ndx << std::endl;
	ndx = 5; Eigen::Vector2i si = Cuben::Fund::ind2sub(dims, ndx);
	std::cout << "Index " << ndx << ", in a [" << dims(0) << " x " << dims(1) << "] matrix, is subindex [" << si(0) << "," << si(1) << "]" << std::endl;
	Eigen::VectorXf rv = Cuben::Fund::initRangeVec(0.1, 0.2, 0.9);
	std::cout << "Range vector: "; Cuben::Fund::printVecTrans(rv); std::cout << std::endl;
	
	// Resizing and other expansions
	Eigen::VectorXf sr = Cuben::Fund::safeResize(rv, 3);
	std::cout << "Range vector resized: "; Cuben::Fund::printVecTrans(sr); std::cout << std::endl;
	Eigen::MatrixXf rm(3,3); rm << 1.2, 3.4, 5.6, 7.8, 9.0, 0.9, 8.7, 6.5, 4.3;
	Eigen::Vector2i newSize; newSize << 3,4;
	Eigen::MatrixXf sm = Cuben::Fund::safeResize(rm, newSize(0), newSize(1));
	std::cout << "Random matrix:" << std::endl << rm << std::endl;
	std::cout << "Resized to [" << newSize(0) << "," << newSize(1) << "], is:" << sm << std::endl;
	Eigen::MatrixXf vdm = Cuben::Fund::vanDerMonde(sr);
	std::cout << "van der Monde of resized range vector:" << std::endl << vdm << std::endl;

	return 0;
}
