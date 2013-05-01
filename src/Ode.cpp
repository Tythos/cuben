/*	Cuben::Ode static object
	Ordinary differential equation algorithms
	Derived from Chapter 6 of Timothy Sauer's 'Numerical Amalysis'
*/

#include "../inc/Cuben.h"

namespace Cuben {
	namespace Ode {
		Eigen::VectorXf euler(float(*dxdt)(float,float), Eigen::VectorXf ti, float x0) {
			int n = ti.rows();
			Eigen::VectorXf xi(n);
			for (int i = 0; i < n; i++) {
				if (i == 0) {
					xi(i) = x0;
				} else {
					xi(i) = xi(i-1) + (ti(i) - ti(i-1)) * dxdt(ti(i-1), xi(i-1));
				}
			}
			return xi;
		}
		
		Eigen::VectorXf trap(float(*dxdt)(float,float), Eigen::VectorXf ti, float x0) {
			int n = ti.rows();
			float sl = 0.0f;
			float sr = 0.0f;
			Eigen::VectorXf xi(n);
			for (int i = 0; i < n; i++) {
				if (i == 0) {
					xi(i) = x0;
				} else {
					sl = dxdt(ti(i-1), xi(i-1));
					sr = dxdt(ti(i), xi(i-1) + (ti(i) - ti(i-1)) * sl);
					xi(i) = xi(i-1) + 0.5f * (sl + sr) * (ti(i) - ti(i-1));
				}
			}
			return xi;
		}
		
		Eigen::VectorXf taylor2nd(float(*dxdt)(float,float), float(*d2xdt2)(float,float), float(*d2xdtdx)(float,float), Eigen::VectorXf ti, float x0) {
			int n = ti.rows();
			float dt = 0.0f;
			Eigen::VectorXf xi(n);
			for (int i = 0; i < n; i++) {
				if (i == 0) {
					xi(i) = x0;
				} else {
					dt = ti(i) - ti(i-1);
					xi(i) = xi(i-1) + dt * dxdt(ti(i-1), xi(i-1)) + 0.5f * dt * dt * (d2xdt2(ti(i-1), xi(i-1)) + d2xdtdx(ti(i-1), xi(i-1)) * dxdt(ti(i-1), xi(i-1)));
				}
			}
			return xi;
		}
		
		Eigen::VectorXf rk4(float(*dxdt)(float,float), Eigen::VectorXf ti, float x0) {
			int n = ti.rows();
			float dt = 0.0f;
			float s1 = 0.0f;
			float s2 = 0.0f;
			float s3 = 0.0f;
			float s4 = 0.0f;
			Eigen::VectorXf xi(n);
			for (int i = 0; i < n; i++) {
				if (i == 0) {
					xi(i) = x0;
				} else {
					dt = ti(i) - ti(i-1);
					s1 = dxdt(ti(i-1), xi(i-1));
					s2 = dxdt(ti(i-1) + 0.5f * dt, xi(i-1) + 0.5f * dt * s1);
					s3 = dxdt(ti(i-1) + 0.5f * dt, xi(i-1) + 0.5f * dt * s2);
					s4 = dxdt(ti(i-1) + dt, xi(i-1) + dt * s3);
					xi(i) = xi(i-1) + (1.0f / 6.0f) * dt * (s1 + 2 * s2 + 2 * s3 + s4);
				}
			}
			return xi;
		}
		
		Eigen::MatrixXf eulerSys(void(*dxdt)(float,Eigen::VectorXf,Eigen::VectorXf&), Eigen::VectorXf ti, Eigen::VectorXf x0) {
			int nSteps = ti.rows();
			int nDims = x0.rows();
			float dt;
			Eigen::MatrixXf xi(nSteps,nDims);
			Eigen::VectorXf dxdtTmp(nDims);
			for (int i = 0; i < nSteps; i++) {
				if (i == 0) {
					xi.row(i) = x0;
				} else {
					dt = ti(i) - ti(i-1);
					Eigen::VectorXf xPrev = xi.row(i-1);
					dxdt(ti(i-1), xPrev, dxdtTmp);
					xi.row(i) = xPrev + dt * dxdtTmp;
				}
			}
			return xi;
		}
		
		Eigen::MatrixXf trapSys(void(*dxdt)(float,Eigen::VectorXf,Eigen::VectorXf&), Eigen::VectorXf ti, Eigen::VectorXf x0) {
			int nSteps = ti.rows();
			int nDims = x0.rows();
			float dt;
			Eigen::VectorXf dxdtLeft(nDims);
			Eigen::VectorXf dxdtRight(nDims);
			Eigen::MatrixXf xi(nSteps,nDims);
			for (int i = 0; i < nSteps; i++) {
				if (i == 0) {
					xi.row(i) = x0;
				} else {
					dt = ti(i) - ti(i-1);
					Eigen::VectorXf xPrev = xi.row(i-1);
					dxdt(ti(i-1), xPrev, dxdtLeft);
					Eigen::VectorXf dx = dt * dxdtLeft;
					Eigen::VectorXf xGuess = xPrev + dx;
					dxdt(ti(i), xGuess, dxdtRight);
					xi.row(i) = xPrev + dt * 0.5f * (dxdtLeft + dxdtRight);
				}
			}
			return xi;
		}
		
		Eigen::MatrixXf midSys(void(*dxdt)(float,Eigen::VectorXf,Eigen::VectorXf&), Eigen::VectorXf ti, Eigen::VectorXf x0) {
			int nSteps = ti.rows();
			int nDims = x0.rows();
			float h = 1.0f;
			Eigen::VectorXf xPrev(nDims);
			Eigen::VectorXf dxdtLeft(nDims);
			Eigen::VectorXf dxdtMid(nDims);
			Eigen::MatrixXf xi(nSteps,nDims);
			for (int i = 0; i < nSteps; i++) {
				if (i == 0) {
					xi.row(i) = x0;
				} else {
					h = ti(i) - ti(i-1);
					xPrev = xi.row(i-1);
					dxdt(ti(i-1), xPrev, dxdtLeft);
					dxdt(ti(i-1) + 0.5f * h, xPrev + 0.5f * h * dxdtLeft, dxdtMid);
					xi.row(i) = xPrev + h * dxdtMid;
				}
			}
			return xi;
		}
		
		Eigen::MatrixXf rk4Sys(void(*dxdt)(float,Eigen::VectorXf,Eigen::VectorXf&), Eigen::VectorXf ti, Eigen::VectorXf x0) {
			int nSteps = ti.rows();
			int nDims = x0.rows();
			float h = 1.0f;
			Eigen::VectorXf s1(nDims);
			Eigen::VectorXf s2(nDims);
			Eigen::VectorXf s3(nDims);
			Eigen::VectorXf s4(nDims);
			Eigen::MatrixXf xi(nSteps,nDims);
			for (int i = 0; i < nSteps; i++) {
				if (i == 0) {
					xi.row(i) = x0;
				} else {
					h = ti(i) - ti(i-1);
					dxdt(ti(i-1), xi.row(i-1), s1);
					dxdt(ti(i-1) + 0.5f * h, xi.row(i-1) + 0.5f * h * s1, s2);
					dxdt(ti(i-1) + 0.5f * h, xi.row(i-1) + 0.5f * h * s2, s3);
					dxdt(ti(i-1) + h, xi.row(i-1) + h * s3, s4);
					xi.row(i) = xi.row(i-1) + (1.0f / 6.0f) * h * (s1 + 2 * s2 + 2 * s2 + s4);
				}
			}
			return xi;
		}
				
		float dxdtSng(float t, float x) {
			return t * x + t * t * t;
		}
		
		void dxdtSys(float t, Eigen::VectorXf x, Eigen::VectorXf &result) {
			float g = 9.81;
			float l = 1.0f;
			result(0) = x(1);
			result(1) = -g * std::sin(x(0)) / l;
		}

		bool test() {
			Eigen::VectorXf ti = Cuben::Fund::initRangeVec(0.0f, 0.1f, 1.0f);
			Eigen::VectorXf xi = rk4(dxdtSng, ti, 1.0f);
			std::cout << "t\tx" << std::endl;
			for (int i = 0; i < ti.rows(); i++) {
				std::cout << ti(i) << "\t" << xi(i) << std::endl;
			}
			std::cout << std::endl;
/*			Eigen::VectorXf x0(2); x0 << M_PI/2.0f,0.0f;
			Eigen::MatrixXf xi = rk4Sys(dxdt, ti, x0);
			std::cout << "t\tx0\tx1" << std::endl;
			for (int i = 0; i < ti.rows(); i++) {
				std::cout << ti(i) << "\t" << xi(i,0) << "\t" << xi(i,1) << std::endl;
			}
			std::cout << std::endl;*/			
			return true;
		}
	}
}
