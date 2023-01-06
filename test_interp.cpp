#include <iostream>
#include <Eigen/Dense>
#include "cuben.h"

int main(int nArgs, char** vArgs) {
	Eigen::VectorXf xi(5); xi << 2,3,5,7,11;
	Eigen::VectorXf yi(5); yi << 1,2,4,8,16;
	for (int i = 0; i < 3; i++) {
		float x = 1.5 + 2 * (i + 1);
		std::cout << "f(" << x << ") = " << Cuben::Interp::lagrange(xi, yi, x) << std::endl;
	}
	return 0;
}
