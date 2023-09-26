#include <iostream>
#include <cmath>
#include "cuben.h"

float f1(float x) {
	return 1.0f / x;
}

float fe(float x) {
	return exp(x);
}

float f2(float x) {
	return log(x);
}

int main(int nArgs, char** vArgs) {
	std::cout << "2-point forward difference of f(x) = 1/x @ x = 2 w/ h = 0.1:" << std::endl;
	std::cout << "    " << Cuben::DiffInt::dfdx_2pfd(f1, 2, 0.1) << std::endl;
	std::cout << "3-point forward difference of f(x) = 1/x @ x = 2 w/ h = 0.1:" << std::endl;
	std::cout << "    " << Cuben::DiffInt::dfdx_3pcd(f1, 2, 0.1) << std::endl;
	std::cout << "h\t2pfd\t3pcd" << std::endl;
	for (int i = 0; i < 8; i++) {
		float h = pow(10,-i);
		std::cout << h << "\t" << Cuben::DiffInt::dfdx_2pfd(fe, 0.0, h) << "\t" << Cuben::DiffInt::dfdx_3pcd(fe, 0.0, h) << std::endl;
	}
	std::cout << "5-point centered second-difference of f(x) = 1/x @ x = 2 w/ h = 0.01:" << std::endl;
	std::cout << "    " << Cuben::DiffInt::d2fdx2_5pcd(f1, 2, 0.1) << std::endl;
	std::cout << "Trapezoidally-integrated f(x) = ln(x), from 1 to 2:" << std::endl;
	std::cout << "    " << Cuben::DiffInt::intfdx_trap(f2, 1.0, 2.0) << std::endl;
	std::cout << "Simpson-integrated f(x) = ln(x), from 1 to 2:" << std::endl;
	std::cout << "    " << Cuben::DiffInt::intfdx_simp(f2, 1.0, 2.0) << std::endl;
	Cuben::DiffInt::test();
	return 0;
}
