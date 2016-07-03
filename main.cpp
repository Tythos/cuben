#include <iostream>
#include "inc/cuben.h"

int main(int nargs, char** vargs) {
	float naan = std::numeric_limits<float>::quiet_NaN();
	std::cout << Cuben::Fund::relEps(16.0);
	return 0;
}
