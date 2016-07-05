#include <iostream>
#include "cuben.h"

float f(float x) {
	return x * x * x + x - 1;
}

float f0(float x) {
	return (1 + 2 * x * x * x) / (1 + 3 * x * x);
}

float dfdx(float x) {
	return 3 * x * x + 1;
}

int main(int nArgs, char** vArgs) {
	float y = Cuben::Equations::bisect(f, 0., 1.);
	std::cout << "Bisection solution: " << y << std::endl;
	y = Cuben::Equations::fpi(f0, 0.5);
	std::cout << "Fixed point iteration: " << y << std::endl;
	y = Cuben::Equations::newt(f, dfdx, -0.7);
	std::cout << "Newton solution: " << y << std::endl;
	y = Cuben::Equations::modNewt(f, dfdx, 1., 1.);
	std::cout << "Modified Newton solution: " << y << std::endl;
	y = Cuben::Equations::secant(f, 0.0, 1.0);
	std::cout << "Secant solution: " << y << std::endl;
	y = Cuben::Equations::regulaFalsi(f, 0., 1.);
	std::cout << "Regula falsi solution: " << y << std::endl;
	y = Cuben::Equations::muller(f, 0., 1.);
	std::cout << "Muller solution: " << y << std::endl;
	y = Cuben::Equations::iqi(f, 0., 1.);
	std::cout << "Inverse quadratic interpolation: " << y << std::endl;
	y = Cuben::Equations::brents(f, 0, 1);
	std::cout << "Brent's method: " << y << std::endl;
	return 0;
}
