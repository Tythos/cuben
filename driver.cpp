#include <iostream>
#include "inc/Cuben.h"

int main(int nargs, char** vargs) {
	std::cout.precision(5);
	if (Cuben::Compress::test()) {
		std::cout << "Test passed" << std::endl;
	} else {
		std::cout << "Test failed" << std::endl;
	}
	return 0;
}
