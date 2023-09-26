#include <iostream>
#include "cuben.h"

float dxdtSng(float t, float x) {
	return t * x + t * t * t;
}

float dxdtStiff(float t, float x) {
	return x + 8.0f * x * x - 9.0f * x * x * x;
}

float dfdxStiff(float t, float x) {
	return 1.0f + 16.0f * x - 27.0f * x * x;
}

void pendSys(float t, Eigen::VectorXf x, Eigen::VectorXf &result) {
	float g = 9.81;
	float l = 1.0f;
	result(0) = x(1);
	result(1) = -g * std::sin(x(0)) / l;
}

void scalarSys(float t, Eigen::VectorXf x, Eigen::VectorXf &result) {
	result(0) = t * x(0) + t * t * t;
}

float dxdtPatho(float t, float x) {
	return -4 * t * t * t * x * x;
}

void dxdtVec(float t, Eigen::VectorXf xi, Eigen::VectorXf& dxidt) {
	dxidt(0) = t * xi(0) + t * t * t;
}

int main(int nArgs, char** vArgs) {
	Cuben::Ode::test(); // Built-in test performs RK23 integration of pendulum ODE
	Eigen::VectorXf ti, xi;
	float dt;
	Eigen::Vector2f tInt;
	Eigen::MatrixXf xij;

	std::cout << std::endl << "Example 6.2, Euler's Method of ty+t^3 on [0,1], y0=0, h=0.1:" << std::endl << std::endl;
	ti = Cuben::Fund::initRangeVec(0.0,0.1,1.0);
	xi = Cuben::Ode::euler(dxdtSng, ti, 1.0);
	Cuben::Fund::printVecSeries(ti, xi, "Time", "Value");

	std::cout << std::endl << "Example 6.9, Euler's Method of -4t3y2 on [-10,0], y0=1/10001, h=1e-[3,4,5]:" << std::endl << std::endl;
	std::cout << "Expected f(0): " << 1.0 << std::endl;
	dt = 1e-3;
	ti = Cuben::Fund::initRangeVec(-10.0, dt, 0.0);
	xi = Cuben::Ode::euler(dxdtPatho, ti, 1.0/10001.0);
	std::cout << "Actual, for step size " << dt << ": " << xi(xi.rows()-1) << std::endl;
	dt = 1e-4;
	ti = Cuben::Fund::initRangeVec(-10.0, dt, 0.0);
	xi = Cuben::Ode::euler(dxdtPatho, ti, 1.0/10001.0);
	std::cout << "Actual, for step size " << dt << ": " << xi(xi.rows()-1) << std::endl;
	dt = 1e-5;
	ti = Cuben::Fund::initRangeVec(-10.0, dt, 0.0);
	xi = Cuben::Ode::euler(dxdtPatho, ti, 1.0/10001.0);
	std::cout << "Actual, for step size " << dt << ": " << xi(xi.rows()-1) << std::endl << std::endl;

	std::cout << std::endl << "Example 6.10, Trapezoid Method of ty+t^3 on [0,1], y0=0, h=0.1:" << std::endl << std::endl;
	ti = Cuben::Fund::initRangeVec(0.0,0.1,1.0);
	xi = Cuben::Ode::trap(dxdtSng, ti, 1.0);
	Cuben::Fund::printVecSeries(ti, xi, "Time", "Value");

	std::cout << std::endl << "Example 6.11, Trapeoid Method of -4t3y2 on [-10,0], y0=1/10001, h=1e-[3,4,5]:" << std::endl << std::endl;
	std::cout << "Expected f(0): " << 1.0 << std::endl;
	dt = 1.5e-3;
	ti = Cuben::Fund::initRangeVec(-10.0, dt, 0.0);
	xi = Cuben::Ode::trap(dxdtPatho, ti, 1.0/10001.0);
	std::cout << "Actual, for step size " << dt << ": " << xi(xi.rows()-1) << std::endl;

	std::cout << std::endl << "Example 6.18, Runge-Kutta 4 of ty+t3 on [0,1], y(0)=1, h=0.1:" << std::endl << std::endl;
	ti = Cuben::Fund::initRangeVec(0.0,0.1,1.0);
	xi = Cuben::Ode::rk4(dxdtSng, ti, 1.0);
	Cuben::Fund::printVecSeries(ti, xi, "Time", "Value");

	std::cout << std::endl << "Example 6.22, Runge-Kutta 4/5 of ty+t3 on [0,1], y(0)=1:" << std::endl << std::endl;
	Cuben::relDiffEqTol = 1e-5;
	tInt << 0.0, 1.0;
	xi = Cuben::Fund::safeResize(xi, 1);
	Cuben::Ode::rk45Sys(dxdtVec, tInt, xi, ti, xij);
	Cuben::Fund::printVecSeries(ti, xij.col(0), "Time", "Value");

	return 0;
}
