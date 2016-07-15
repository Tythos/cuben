/*	Cuben::Ode static object
	Ordinary differential equation algorithms
	Derived from Chapter 6 of Timothy Sauer's 'Numerical Amalysis'
*/

#include "cuben.h"

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

		void rk23Sys(void(*dxdt)(float,Eigen::VectorXf,Eigen::VectorXf&), Eigen::Vector2f tInt, Eigen::VectorXf x0, Eigen::VectorXf& ti, Eigen::MatrixXf& xi) {
			int nDims = x0.rows();
			int n = 1;
			int order = 3;
			float relErr = 0.0f;
			float absErr = 0.0f;
			float minStepSize = 16.0f * Cuben::Fund::relEps(tInt(1));
			float h = std::sqrt(minStepSize * (tInt(1) - tInt(0)));
			float maxStepSize = h;
			Eigen::VectorXf s1(nDims);
			Eigen::VectorXf s2(nDims);
			Eigen::VectorXf s3(nDims);
			Eigen::VectorXf o2(nDims);
			Eigen::VectorXf o3(nDims);
			Eigen::VectorXf xPrev(nDims);
			ti = Eigen::VectorXf(1);
			xi = Eigen::MatrixXf(1,nDims);
			xi.row(0) = x0;
			ti(0) = tInt(0);
			
			// Iterate on variable step size, using difference between RK3 and RK4 to compute error
			while (ti(n - 1) < tInt(1)) {
				xPrev = xi.row(n - 1);
				dxdt(ti(n - 1), xPrev, s1);
				dxdt(ti(n - 1) + h, xPrev + h * s1, s2);
				dxdt(ti(n - 1) + 0.5 * h, xPrev + 0.5f * h * (s1 + s2), s3);
				o2 = xPrev + 0.5f * h * (s1 + s2);
				o3 = xPrev + (1.0f / 6.0f) * h * (s1 + 4 * s3 + s2);
				absErr = std::max((1.0f / 3.0f) * h * (s1 - 2.0f * s3 + s2).norm(), (o3 - o2).norm());
				relErr = absErr / o2.norm();
				if (relErr < Cuben::relDiffEqTol) {
					// Update state with o3, compute h for next step
					ti = Cuben::Fund::safeResize(ti, n + 1);
					xi = Cuben::Fund::safeResize(xi, n + 1, nDims);
					ti(n) = ti(n - 1) + h;
					xi.row(n) = o3;
					maxStepSize = std::min(tInt(1) - ti(n), 10.0f * h);
					h = std::max(minStepSize, std::min(maxStepSize, (0.8f * std::pow(Cuben::relDiffEqTol / relErr, 1.0f / (order + 1.0f)))));
					minStepSize = 16.0f * Cuben::Fund::relEps(ti(n));
					n++;
				} else {
					// Try again with half step size
					h *= 0.5f;
					if (h < minStepSize) { throw Cuben::xBelowMinStepSize(); }
				}
			}
		}

		void bs23Sys(void(*dxdt)(float,Eigen::VectorXf,Eigen::VectorXf&), Eigen::Vector2f tInt, Eigen::VectorXf x0, Eigen::VectorXf& ti, Eigen::MatrixXf& xi) {
			int nDims = x0.rows();
			int n = 1;
			int order = 3;
			float relErr = 0.0f;
			float absErr = 0.0f;
			float minStepSize = 16.0f * Cuben::Fund::relEps(tInt(1));
			float h = std::sqrt(minStepSize * (tInt(1) - tInt(0)));
			float maxStepSize = h;
			Eigen::VectorXf s1(nDims);
			Eigen::VectorXf s2(nDims);
			Eigen::VectorXf s3(nDims);
			Eigen::VectorXf s4(nDims);
			Eigen::VectorXf o2(nDims);
			Eigen::VectorXf o3(nDims);
			Eigen::VectorXf xPrev(nDims);
			ti = Eigen::VectorXf(1);
			xi = Eigen::MatrixXf(1,nDims);
			xi.row(0) = x0;
			ti(0) = tInt(0);
			
			// Proceed with each new step size, recomputing for smaller step if necessary
			while (ti(n - 1) < tInt(1)) {
				xPrev = xi.row(n - 1);
				dxdt(ti(n - 1), xPrev, s1);
				dxdt(ti(n - 1) + 0.5f * h, xPrev + 0.5f * h * s1, s2);
				dxdt(ti(n - 1) + 0.75f * h, xPrev + 0.75f * h * s2, s3);
				o3 = xPrev + (1.0f / 9.0f) * h * (2.0f * s1 + 3.0f * s2 + 4.0f * s3);
				dxdt(ti(n - 1) + h, o3, s4);
				o2 = xPrev + (1.0f / 24.0f)  * h * (7.0f * s1 + 6.0f * s2 + 8.0f * s3 + 3.0f * s4);
				absErr = std::max((1.0f / 72.0f) * h * (-5.0f * s1 + 6.0f * s2 + 8.0f * s3 - 9.0f * s4).norm(), (o3 - o2).norm());
				relErr = absErr / o2.norm();
				if (relErr < Cuben::relDiffEqTol) {
					// Update state with o3, compute h for next step
					ti = Cuben::Fund::safeResize(ti, n + 1);
					xi = Cuben::Fund::safeResize(xi, n + 1, nDims);
					ti(n) = ti(n - 1) + h;
					xi.row(n) = o3;
					maxStepSize = std::min(tInt(1) - ti(n), 10.0f * h);
					h = std::max(minStepSize, std::min(maxStepSize, (0.8f * std::pow(Cuben::relDiffEqTol / relErr, 1.0f / (order + 1.0f)))));
					minStepSize = 16.0f * Cuben::Fund::relEps(ti(n));
					n++;
				} else {
					// Try again with half step size
					h *= 0.5f;
					if (h < minStepSize) { throw Cuben::xBelowMinStepSize(); }
				}
			}
		}

		void rk45Sys(void(*dxdt)(float,Eigen::VectorXf,Eigen::VectorXf&), Eigen::Vector2f tInt, Eigen::VectorXf x0, Eigen::VectorXf& ti, Eigen::MatrixXf& xi) {
			int nDims = x0.rows();
			int n = 1;
			int order = 5;
			float relErr = 0.0f;
			float absErr = 0.0f;
			float minStepSize = 16.0f * Cuben::Fund::relEps(tInt(1));
			float h = std::sqrt(minStepSize * (tInt(1) - tInt(0)));
			float maxStepSize = h;
			Eigen::VectorXf s1(nDims);
			Eigen::VectorXf s2(nDims);
			Eigen::VectorXf s3(nDims);
			Eigen::VectorXf s4(nDims);
			Eigen::VectorXf s5(nDims);
			Eigen::VectorXf s6(nDims);
			Eigen::VectorXf o4(nDims);
			Eigen::VectorXf o5(nDims);
			Eigen::VectorXf xPrev(nDims);
			ti = Eigen::VectorXf(1);
			xi = Eigen::MatrixXf(1,nDims);
			xi.row(0) = x0;
			ti(0) = tInt(0);
			while (ti(n - 1) < tInt(1)) {
				xPrev = xi.row(n - 1);
				dxdt(ti(n - 1), xPrev, s1);
				dxdt(ti(n - 1) + 0.25f * h, xPrev + 0.25 * h * s1, s2);
				dxdt(ti(n - 1) + 0.125f * h, xPrev + 0.09375f * h * s1 + 0.28125f * h * s2, s3);
				dxdt(ti(n - 1) + (12.0f/13.0f) * h, xPrev + (1932.0f/2197.0f) * h * s1 - (7200.0f/2197.0f) * h * s2 + (7296.0f/2197.0f) * h * s3, s4);
				dxdt(ti(n - 1) + h, xPrev + (439.0f/216.0f) * h * s1 - 8.0f * h * s2 + (3680.0f/513.0f) * h * s3 - (845.0f/4104.0f) * h * s4, s5);
				dxdt(ti(n - 1) + 0.5f * h, xPrev - (8.0f/27.0f) * h * s1 + 2.0f * h * s2 - (4544.0f/2565.0f) * h * s3 + (1859.0f/4104.0f) * h * s4 - (11.0f/40.0f) * h * s5, s6);
				o4 = xPrev + h * ((25.0f/216.0f) * s1 + (1408.0f/2565.0f) * s3 + (2197.0f/4104.0f) * s4 - 0.2f * s5);
				o5 = xPrev + h * ((15.0f/135.0f) * s1 + (6656.0f/12825.0f) * s3 + (28561.0f/56430.0f) * s4 - 0.18f * s5 + (2.0f/55.0f) * s6);
				absErr = std::max(h * ((1.0f/360.0f) * s1 - (128.0f/4275.0f) * s3 - (2197.0f/75240.0f) * s4 + 0.02f * s5 + (2.0f/55.0f) * s6).norm(), (o4 - o5).norm());
				relErr = absErr / o4.norm();
				if (relErr < Cuben::relDiffEqTol) {
					// Update state with o5, compute h for next step
					ti = Cuben::Fund::safeResize(ti, n + 1);
					xi = Cuben::Fund::safeResize(xi, n + 1, nDims);
					ti(n) = ti(n - 1) + h;
					xi.row(n) = o5;
					maxStepSize = std::min(tInt(1) - ti(n), 10.0f * h);
					h = std::max(minStepSize, std::min(maxStepSize, (0.8f * std::pow(Cuben::relDiffEqTol / relErr, 1.0f / (order + 1.0f)))));
					minStepSize = 16.0f * Cuben::Fund::relEps(ti(n));
					n++;
				} else {
					// Try again with half the step size
					h *= 0.5f;
					if (h < minStepSize) { throw Cuben::xBelowMinStepSize(); }
				}
			}
		}
		
		void dp45Sys(void(*dxdt)(float,Eigen::VectorXf,Eigen::VectorXf&), Eigen::Vector2f tInt, Eigen::VectorXf x0, Eigen::VectorXf& ti, Eigen::MatrixXf& xi) {
			int nDims = x0.rows();
			int n = 1;
			int order = 5;
			float relErr = 0.0f;
			float absErr = 0.0f;
			float minStepSize = 16.0f * Cuben::Fund::relEps(tInt(1));
			float h = std::sqrt(minStepSize * (tInt(1) - tInt(0)));
			float maxStepSize = h;
			Eigen::VectorXf s1(nDims);
			Eigen::VectorXf s2(nDims);
			Eigen::VectorXf s3(nDims);
			Eigen::VectorXf s4(nDims);
			Eigen::VectorXf s5(nDims);
			Eigen::VectorXf s6(nDims);
			Eigen::VectorXf s7(nDims);
			Eigen::VectorXf o4(nDims);
			Eigen::VectorXf o5(nDims);
			Eigen::VectorXf xPrev(nDims);
			ti = Eigen::VectorXf(1);
			xi = Eigen::MatrixXf(1,nDims);
			xi.row(0) = x0;
			ti(0) = tInt(0);
			
			// Proceed with each new step size, recomputing for smaller step if necessary
			while (ti(n - 1) < tInt(1)) {
				xPrev = xi.row(n - 1);
				dxdt(ti(n - 1), xPrev, s1);
				dxdt(ti(n - 1) + 0.2f * h, xPrev + 0.2 * h * s1, s2);
				dxdt(ti(n - 1) + 0.3f * h, xPrev + 0.075f * h * s1 + 0.225f * h * s2, s3);
				dxdt(ti(n - 1) + 0.8f * h, xPrev + (44.0f/45.0f) * h * s1 - (56.0f/15.0f) * h * s2 + (32.0f/9.0f) * h * s3, s4);
				dxdt(ti(n - 1) + (8.0f/9.0f) * h, xPrev + (19372.0f/6561.0f) * h * s1 - (25360.0f/2187.0f) * h * s2 + (64448.0f/6561.0f) * h * s3 - (212.0f/729.0f) * h * s4, s5);
				dxdt(ti(n - 1) + h, xPrev + (9017.0f/3168.0f) * h * s1 - (355.0f/33.0f) * h * s2 + (46732.0f/5247.0f) * h * s3 + (49.0f/176.0f) * h * s4 - (5103.0f/18656.0f) * h * s5, s6);
				o5 = xPrev + h * ((35.0f/384.0f) * s1 + (500.0f/1113.0f) * s3 + (125.0f/192.0f) * s4 - (2187.0f/6784.0f) * s5 + (11.0f/84.0f) * s6);
				dxdt(ti(n - 1) + h, o5, s7);
				o4 = xPrev + h * ((5179.0f/57600.0f) * s1 + (7571.0f/16695.0f) * s3 + (393.0f/640.0f) * s4 - (92097.0f/339200.0f) * s5 + (187.0f/2100.0f) * s6 + 0.025f * s7);
				absErr = std::max(h * ((71.0f/57600.0f) * s1 - (71.0f/16695.0f) * s3 + (71.0f/1920.0f) * s4 - (17253.0f/339200.0f) * s5 + (22.0f/525.0f) * s6 - 0.025f * s7).norm(), (o4 - o5).norm());
				relErr = absErr / o4.norm();
				if (relErr < Cuben::relDiffEqTol) {
					// Update state with o5, compute h for next step
					ti = Cuben::Fund::safeResize(ti, n + 1);
					xi = Cuben::Fund::safeResize(xi, n + 1, nDims);
					ti(n) = ti(n - 1) + h;
					xi.row(n) = o5;
					maxStepSize = std::min(tInt(1) - ti(n), 10.0f * h);
					h = std::max(minStepSize, std::min(maxStepSize, (0.8f * std::pow(Cuben::relDiffEqTol / relErr, 1.0f / (order + 1.0f)))));
					minStepSize = 16.0f * Cuben::Fund::relEps(ti(n));
					n++;
				} else {
					// Try again with half the step size
					h *= 0.5f;
					if (h < minStepSize) { throw Cuben::xBelowMinStepSize(); }
				}
			}
		}
		
		Eigen::VectorXf impEuler(float(*fInd)(float,float), float(*fImp)(float,float), Eigen::VectorXf ti, float x0) {
			int n = ti.rows();
			float dt = 0.0f;
			float ox = 0.0f;
			float dx = 0.0f;
			int nn = 0;
			Eigen::VectorXf xi(n);
			for (int i = 0; i < n; i++) {
				if (i == 0) {
					xi(i) = x0;
				} else {
					// Using initial guess of xi(i-1), perform a Newtonian iteration to converge on xi(i)
					xi(i) = xi(i-1);
					dt = ti(i) - ti(i-1);
					nn = 0;
					do {
						nn++;
						ox = xi(i);
						dx = ox - xi(i-1);
						xi(i) = ox - (dt * fInd(ti(i), ox) - dx) / (dt * fImp(ti(i), ox) - 1.0f);
					} while (nn < Cuben::iterLimit && std::abs(xi(i) - ox) > Cuben::iterTol);
				}
			}
			return xi;
		}
		
		Eigen::VectorXf impTrap(float(*fInd)(float,float), float(*fImp)(float,float), Eigen::VectorXf ti, float x0) {
			int n = ti.rows();
			float dt = 0.0f;
			float ox = 0.0f;
			float dx = 0.0f;
			int nn = 0;
			Eigen::VectorXf xi(n);
			for (int i = 0; i < n; i++) {
				if (i == 0) {
					xi(i) = x0;
				} else {
					// Using initial guess of xi(i-1), perform a Newtonian iteration to converge on xi(i)
					xi(i) = xi(i-1);
					dt = ti(i) - ti(i-1);
					nn = 0;
					do {
						nn++;
						ox = xi(i);
						dx = ox - xi(i-1);
						xi(i) = ox - (0.5f * dt * (fInd(ti(i), ox) + fInd(ti(i-1), xi(i-1))) - dx) / (0.5f * dt * (fImp(ti(i), ox) + fImp(ti(i-1), xi(i-1))) - 1.0f);
					} while (nn < Cuben::iterLimit && std::abs(xi(i) - ox) > Cuben::iterTol);
				}
			}
			return xi;
		}
		
		Eigen::VectorXf modab2s(float(*dxdt)(float,float), Eigen::VectorXf ti, float x0) {
			int n = ti.rows();
			float dt = 0.0f;
			Eigen::VectorXf xi(n);
			for (int i = 0; i < n; i++) {
				if (i == 0) {
					xi(i) = x0;
				} else if (i == 1) {
					// Fall back to trapezoid
					dt = ti(i) - ti(i-1);
					xi(i) = xi(i-1) + dt * 0.5f * (dxdt(ti(i-1), xi(i-1)) + dxdt(ti(i), xi(i-1) + dt * dxdt(ti(i-1), xi(i-1))));
				} else {
					dt = 0.5f * (ti(i) - ti(i-2));
					xi(i) = xi(i-1) + dt * dt * (1.5f * dxdt(ti(i-1),xi(i-1)) / (ti(i) - ti(i-1)) - 0.5f * dxdt(ti(i-2),xi(i-2)) / (ti(i-1) - ti(i-2)));
				}
			}
			return xi;
		}
		
		Eigen::VectorXf modab3s(float(*dxdt)(float,float), Eigen::VectorXf ti, float x0) {
			int n = ti.rows();
			float dt = 0.0f;
			Eigen::VectorXf xi(n);
			for (int i = 0; i < n; i++) {
				if (i == 0) {
					xi(i) = x0;
				} else if (i == 1) {
					// Fall back to trapezoid
					dt = ti(i) - ti(i-1);
					xi(i) = xi(i-1) + dt * 0.5f * (dxdt(ti(i-1), xi(i-1)) + dxdt(ti(i), xi(i-1) + dt * dxdt(ti(i-1), xi(i-1))));
				} else if (i == 2) {
					// Fall back to modab2s
					dt = 0.5f * (ti(i) - ti(i-2));
					xi(i) = xi(i-1) + dt * dt * (1.5f * dxdt(ti(i-1),xi(i-1)) / (ti(i) - ti(i-1)) - 0.5f * dxdt(ti(i-2),xi(i-2)) / (ti(i-1) - ti(i-2)));
				} else {
					dt = (1.0f/3.0f) * (ti(i) - ti(i-3));
					xi(i) = xi(i-1) + (1.0f/12.0f) * dt * dt * (23.0f * dxdt(ti(i-1),xi(i-1)) / (ti(i) - ti(i-1)) - 16.0f * dxdt(ti(i-2),xi(i-2)) / (ti(i-1) - ti(i-2)) + 5.0f * dxdt(ti(i-3),xi(i-3)) / (ti(i-2) - ti(i-3)));
				}
			}
			return xi;
		}
		
		Eigen::VectorXf modab4s(float(*dxdt)(float,float), Eigen::VectorXf ti, float x0) {
			int n = ti.rows();
			float dt = 0.0f;
			Eigen::VectorXf xi(n);
			for (int i = 0; i < n; i++) {
				if (i == 0) {
					xi(i) = x0;
				} else if (i == 1) {
					// Fall back to trapezoid
					dt = ti(i) - ti(i-1);
					xi(i) = xi(i-1) + dt * 0.5f * (dxdt(ti(i-1), xi(i-1)) + dxdt(ti(i), xi(i-1) + dt * dxdt(ti(i-1), xi(i-1))));
				} else if (i == 2) {
					// Fall back to modab2s
					dt = 0.5f * (ti(i) - ti(i-2));
					xi(i) = xi(i-1) + dt * dt * (1.5f * dxdt(ti(i-1),xi(i-1)) / (ti(i) - ti(i-1)) - 0.5f * dxdt(ti(i-2),xi(i-2)) / (ti(i-1) - ti(i-2)));
				} else if (i == 3) {
					// Fall back to modab3s
					dt = (1.0f/3.0f) * (ti(i) - ti(i-3));
					xi(i) = xi(i-1) + (1.0f/12.0f) * dt * dt * (23.0f * dxdt(ti(i-1),xi(i-1)) / (ti(i) - ti(i-1)) - 16.0f * dxdt(ti(i-2),xi(i-2)) / (ti(i-1) - ti(i-2)) + 5.0f * dxdt(ti(i-3),xi(i-3)) / (ti(i-2) - ti(i-3)));
				} else {
					dt = 0.25f * (ti(i) - ti(i-4));
					xi(i) = xi(i-1) + (1.0f/24.0f) * dt * dt * (55.0f * dxdt(ti(i-1),xi(i-1)) / (ti(i) - ti(i-1)) - 59.0f * dxdt(ti(i-2),xi(i-2)) / (ti(i-1) - ti(i-2)) + 37.0f * dxdt(ti(i-3),xi(i-3)) / (ti(i-2) - ti(i-3)) - 9.0f * dxdt(ti(i-4),xi(i-4)) / (ti(i-3) - ti(i-4)));
				}
			}
			return xi;
		}
		
		Eigen::VectorXf modam2s(float(*fInd)(float,float), float(*fImp)(float,float), Eigen::VectorXf ti, float x0) {
			int n = ti.rows();
			float dt = 0.0f;
			float ox = 0.0f;
			float dx = 0.0f;
			int nn = 0;
			Eigen::VectorXf xi(n);
			for (int i = 0; i < n; i++) {
				if (i == 0) {
					xi(i) = x0;
				} else if (i == 1) {
					// Fall back to implicit trapezoid method for first step
					xi(i) = xi(i-1);
					dt = ti(i) - ti(i-1);
					nn = 0;
					do {
						nn++;
						ox = xi(i);
						dx = ox - xi(i-1);
						xi(i) = ox - (0.5f * dt * (fInd(ti(i), ox) + fInd(ti(i-1), xi(i-1))) - dx) / (0.5f * dt * (fImp(ti(i), ox) + fImp(ti(i-1), xi(i-1))) - 1.0f);
					} while (nn < Cuben::iterLimit && std::abs(xi(i) - ox) > Cuben::iterTol);
				} else if (i == 2) {
					// Same alg as i = 3, but ti(i-2) - t(i-3) replaced with avg step size
					xi(i) = xi(i-1);
					dt = (1.0f/2.0f) * (ti(i) - ti(i-2));
					nn = 0;
					do {
						nn++;
						ox = xi(i);
						dx = ox - xi(i-1);
						xi(i) = ox - ((1.0f/12.0f) * dt * dt * (5.0f * fInd(ti(i), ox) / (ti(i) - ti(i-1)) + 8.0f * fInd(ti(i-1), xi(i-1)) / (ti(i-1) - ti(i-2)) - fInd(ti(i-2), xi(i-2)) / dt) - dx) / ((1.0f/12.0f) * dt * dt * (5.0f * fImp(ti(i), ox) / (ti(i) - ti(i-1)) + 8.0f * fImp(ti(i-1), xi(i-1)) / (ti(i-1) - ti(i-2)) - fImp(ti(i-2), xi(i-2)) / dt) - 1.0f);
					} while (nn < Cuben::iterLimit && std::abs(xi(i) - ox) > Cuben::iterTol);
				} else {
					// Using initial guess of xi(i-1), perform a Newtonian iteration to converge on xi(i)
					xi(i) = xi(i-1);
					dt = (1.0f/3.0f) * (ti(i) - ti(i-3));
					nn = 0;
					do {
						nn++;
						ox = xi(i);
						dx = ox - xi(i-1);
						xi(i) = ox - ((1.0f/12.0f) * dt * dt * (5.0f * fInd(ti(i), ox) / (ti(i) - ti(i-1)) + 8.0f * fInd(ti(i-1), xi(i-1)) / (ti(i-1) - ti(i-2)) - fInd(ti(i-2), xi(i-2)) / (ti(i-2) - ti(i-3))) - dx) / ((1.0f/12.0f) * dt * dt * (5.0f * fImp(ti(i), ox) / (ti(i) - ti(i-1)) + 8.0f * fImp(ti(i-1), xi(i-1)) / (ti(i-1) - ti(i-2)) - fImp(ti(i-2), xi(i-2)) / (ti(i-2) - ti(i-3))) - 1.0f);
					} while (nn < Cuben::iterLimit && std::abs(xi(i) - ox) > Cuben::iterTol);
				}
			}
			return xi;
		}
		
		Eigen::VectorXf modms2s(float(*fInd)(float,float), float(*fImp)(float,float), Eigen::VectorXf ti, float x0) {
			// Modified Milne-Simpson method--NOTE that this is only weakly stable,
			// and is implemented here for academic purposes only.
			int n = ti.rows();
			float dt = 0.0f;
			float ox = 0.0f;
			float dx = 0.0f;
			int nn = 0;
			Eigen::VectorXf xi(n);
			for (int i = 0; i < n; i++) {
				if (i == 0) {
					xi(i) = x0;
				} else if (i == 1) {
					// Fall back to implicit trapezoid method for first step
					xi(i) = xi(i-1);
					dt = ti(i) - ti(i-1);
					nn = 0;
					do {
						nn++;
						ox = xi(i);
						dx = ox - xi(i-1);
						xi(i) = ox - (0.5f * dt * (fInd(ti(i), ox) + fInd(ti(i-1), xi(i-1))) - dx) / (0.5f * dt * (fImp(ti(i), ox) + fImp(ti(i-1), xi(i-1))) - 1.0f);
					} while (nn < Cuben::iterLimit && std::abs(xi(i) - ox) > Cuben::iterTol);
				} else if (i == 2) {
					// Same as i = 3, only replace ti(i-2) - ti(i-3) with dt
					xi(i) = xi(i-1);
					dt = (1.0f/2.0f) * (ti(i) - ti(i-2));
					nn = 0;
					do {
						nn++;
						ox = xi(i);
						dx = ox - xi(i-2);
						xi(i) = ox - ((1.0f/3.0f) * dt * dt * (fInd(ti(i), ox) / (ti(i) - ti(i-1)) + 4.0f * fInd(ti(i-1), xi(i-1)) / (ti(i-1) - ti(i-2)) + fInd(ti(i-2), xi(i-2)) / dt) - dx) / ((1.0f/3.0f) * dt * dt * (fImp(ti(i), ox) / (ti(i) - ti(i-1)) + 4.0f * fImp(ti(i-1), xi(i-1)) / (ti(i-1) - ti(i-2)) + fImp(ti(i-2), xi(i-2)) / dt) - 1.0f);
					} while (nn < Cuben::iterLimit && std::abs(xi(i) - ox) > Cuben::iterTol);
				} else {
					// Using initial guess of xi(i-1), perform a Newtonian iteration to converge on xi(i)
					xi(i) = xi(i-1);
					dt = (1.0f/3.0f) * (ti(i) - ti(i-3));
					nn = 0;
					do {
						nn++;
						ox = xi(i);
						dx = ox - xi(i-2);
						xi(i) = ox - ((1.0f/3.0f) * dt * dt * (fInd(ti(i), ox) / (ti(i) - ti(i-1)) + 4.0f * fInd(ti(i-1), xi(i-1)) / (ti(i-1) - ti(i-2)) + fInd(ti(i-2), xi(i-2)) / (ti(i-2) - ti(i-3))) - dx) / ((1.0f/3.0f) * dt * dt * (fImp(ti(i), ox) / (ti(i) - ti(i-1)) + 4.0f * fImp(ti(i-1), xi(i-1)) / (ti(i-1) - ti(i-2)) + fImp(ti(i-2), xi(i-2)) / (ti(i-2) - ti(i-3))) - 1.0f);
					} while (nn < Cuben::iterLimit && std::abs(xi(i) - ox) > Cuben::iterTol);
				}
			}
			return xi;
		}

		Eigen::VectorXf modam3s(float(*fInd)(float,float), float(*fImp)(float,float), Eigen::VectorXf ti, float x0) {
			// Modified Adams-Moulton 3-step method, adjusted for non-uniform independent variable spacing
			int n = ti.rows();
			int nn = 0;
			float dt = 0.0f;
			float ox = 0.0f;
			float dx = 0.0f;
			Eigen::VectorXf xi(n);
			for (int i = 0; i < n; i++) {
				if (i == 0) {
					xi(i) = x0;
				} else if (i == 1) {
					// Fall back to implicit trapezoid method for first step
					xi(i) = xi(i-1);
					dt = ti(i) - ti(i-1);
					nn = 0;
					do {
						nn++;
						ox = xi(i);
						dx = ox - xi(i-1);
						xi(i) = ox - (0.5f * dt * (fInd(ti(i), ox) + fInd(ti(i-1), xi(i-1))) - dx) / (0.5f * dt * (fImp(ti(i), ox) + fImp(ti(i-1), xi(i-1))) - 1.0f);
					} while (nn < Cuben::iterLimit && std::abs(xi(i) - ox) > Cuben::iterTol);
				} else if (i == 2) {
					// Fall back to 2-step method
					xi(i) = xi(i-1);
					dt = (1.0f/2.0f) * (ti(i) - ti(i-2));
					nn = 0;
					do {
						nn++;
						ox = xi(i);
						dx = ox - xi(i-1);
						xi(i) = ox - ((1.0f/12.0f) * dt * dt * (5.0f * fInd(ti(i), ox) / (ti(i) - ti(i-1)) + 8.0f * fInd(ti(i-1), xi(i-1)) / (ti(i-1) - ti(i-2)) - fInd(ti(i-2), xi(i-2)) / dt) - dx) / ((1.0f/12.0f) * dt * dt * (5.0f * fImp(ti(i), ox) / (ti(i) - ti(i-1)) + 8.0f * fImp(ti(i-1), xi(i-1)) / (ti(i-1) - ti(i-2)) - fImp(ti(i-2), xi(i-2)) / dt) - 1.0f);
					} while (nn < Cuben::iterLimit && std::abs(xi(i) - ox) > Cuben::iterTol);
				} else if (i == 3) {
					// Same as i >= 4, but ti(i-3) - ti(i-4) is replaced by average dt
					xi(i) = xi(i-1);
					dt = (1.0f/3.0f) * (ti(i) - ti(i-3));
					nn = 0;
					do {
						nn++;
						ox = xi(i);
						dx = ox - xi(i-1);
						xi(i) = ox - ((1.0f/24.0f) * dt * dt * (9.0f * fInd(ti(i), ox) / (ti(i) - ti(i-1)) + 19.0f * fInd(ti(i-1), xi(i-1)) / (ti(i-1) - ti(i-2)) - 5.0f * fInd(ti(i-2), xi(i-2)) / (ti(i-2) - ti(i-3)) + fInd(ti(i-3), xi(i-3)) / dt) - dx) / ((1.0f/24.0f) * dt * dt * (9.0f * fImp(ti(i), ox) / (ti(i) - ti(i-1)) + 19.0f * fImp(ti(i-1), xi(i-1)) / (ti(i-1) - ti(i-2)) - 5.0f * fImp(ti(i-2), xi(i-2)) / (ti(i-2) - ti(i-3)) + fImp(ti(i-3), xi(i-3)) / dt) - 1.0f);
					} while(nn < Cuben::iterLimit && std::abs(xi(i) - ox) > Cuben::iterTol);
				} else {
					xi(i) = xi(i-1);
					dt = (1.0f/4.0f) * (ti(i) - ti(i-4));
					nn = 0;
					do {
						nn++;
						ox = xi(i);
						dx = ox - xi(i-1);
						xi(i) = ox - ((1.0f/24.0f) * dt * dt * (9.0f * fInd(ti(i), ox) / (ti(i) - ti(i-1)) + 19.0f * fInd(ti(i-1), xi(i-1)) / (ti(i-1) - ti(i-2)) - 5.0f * fInd(ti(i-2), xi(i-2)) / (ti(i-2) - ti(i-3)) + fInd(ti(i-3), xi(i-3)) / (ti(i-3) - ti(i-4))) - dx) / ((1.0f/24.0f) * dt * dt * (9.0f * fImp(ti(i), ox) / (ti(i) - ti(i-1)) + 19.0f * fImp(ti(i-1), xi(i-1)) / (ti(i-1) - ti(i-2)) - 5.0f * fImp(ti(i-2), xi(i-2)) / (ti(i-2) - ti(i-3)) + fImp(ti(i-3), xi(i-3)) / (ti(i-3) - ti(i-4))) - 1.0f);
					} while(nn < Cuben::iterLimit && std::abs(xi(i) - ox) > Cuben::iterTol);
				}
			}
			return xi;
		}
		
		Eigen::VectorXf modam4s(float(*fInd)(float,float), float(*fImp)(float,float), Eigen::VectorXf ti, float x0) {
			// Modified Adams-Moulton 4-step method, adjusted for non-uniform independent variable spacing
			int n = ti.rows();
			int nn = 0;
			float dt = 0.0f;
			float ox = 0.0f;
			float dx = 0.0f;
			Eigen::VectorXf xi(n);
			for (int i = 0; i < n; i++) {
				if (i == 0) {
					xi(i) = x0;
				} else if (i == 1) {
					// Fall back to implicit trapezoid method for first step
					xi(i) = xi(i-1);
					dt = ti(i) - ti(i-1);
					nn = 0;
					do {
						nn++;
						ox = xi(i);
						dx = ox - xi(i-1);
						xi(i) = ox - (0.5f * dt * (fInd(ti(i), ox) + fInd(ti(i-1), xi(i-1))) - dx) / (0.5f * dt * (fImp(ti(i), ox) + fImp(ti(i-1), xi(i-1))) - 1.0f);
					} while (nn < Cuben::iterLimit && std::abs(xi(i) - ox) > Cuben::iterTol);
				} else if (i == 2) {
					// Fall back to 2-step method
					xi(i) = xi(i-1);
					dt = (1.0f/2.0f) * (ti(i) - ti(i-2));
					nn = 0;
					do {
						nn++;
						ox = xi(i);
						dx = ox - xi(i-1);
						xi(i) = ox - ((1.0f/12.0f) * dt * dt * (5.0f * fInd(ti(i), ox) / (ti(i) - ti(i-1)) + 8.0f * fInd(ti(i-1), xi(i-1)) / (ti(i-1) - ti(i-2)) - fInd(ti(i-2), xi(i-2)) / dt) - dx) / ((1.0f/12.0f) * dt * dt * (5.0f * fImp(ti(i), ox) / (ti(i) - ti(i-1)) + 8.0f * fImp(ti(i-1), xi(i-1)) / (ti(i-1) - ti(i-2)) - fImp(ti(i-2), xi(i-2)) / dt) - 1.0f);
					} while (nn < Cuben::iterLimit && std::abs(xi(i) - ox) > Cuben::iterTol);
				} else if (i == 3) {
					// Fall back to 3-step method
					xi(i) = xi(i-1);
					dt = (1.0f/3.0f) * (ti(i) - ti(i-3));
					nn = 0;
					do {
						nn++;
						ox = xi(i);
						dx = ox - xi(i-1);
						xi(i) = ox - ((1.0f/24.0f) * dt * dt * (9.0f * fInd(ti(i), ox) / (ti(i) - ti(i-1)) + 19.0f * fInd(ti(i-1), xi(i-1)) / (ti(i-1) - ti(i-2)) - 5.0f * fInd(ti(i-2), xi(i-2)) / (ti(i-2) - ti(i-3)) + fInd(ti(i-3), xi(i-3)) / dt) - dx) / ((1.0f/24.0f) * dt * dt * (9.0f * fImp(ti(i), ox) / (ti(i) - ti(i-1)) + 19.0f * fImp(ti(i-1), xi(i-1)) / (ti(i-1) - ti(i-2)) - 5.0f * fImp(ti(i-2), xi(i-2)) / (ti(i-2) - ti(i-3)) + fImp(ti(i-3), xi(i-3)) / dt) - 1.0f);
					} while(nn < Cuben::iterLimit && std::abs(xi(i) - ox) > Cuben::iterTol);
				} else if (i == 4) {
					// Same as i >= 4, but ti(i-3) - ti(i-4) is replaced by average dt
					xi(i) = xi(i-1);
					dt = (1.0f/4.0f) * (ti(i) - ti(i-4));
					nn = 0;
					do {
						nn++;
						ox = xi(i);
						dx = ox - xi(i-1);
						xi(i) = ox - ((1.0f/720.0f) * dt * dt * (251.0f * fInd(ti(i), ox) / (ti(i) - ti(i-1)) + 646.0f * fInd(ti(i-1), xi(i-1)) / (ti(i-1) - ti(i-2)) - 264.0f * fInd(ti(i-2), xi(i-2)) / (ti(i-2) - ti(i-3)) + 106.0f * fInd(ti(i-3), xi(i-3)) / (ti(i-3) - ti(i-4)) - 19.0f * fInd(ti(i-4), xi(i-4)) / dt) - dx) / ((1.0f/720.0f) * dt * dt * (251.0f * fImp(ti(i), ox) / (ti(i) - ti(i-1)) + 646.0f * fImp(ti(i-1), xi(i-1)) / (ti(i-1) - ti(i-2)) - 264.0f * fImp(ti(i-2), xi(i-2)) / (ti(i-2) - ti(i-3)) + 106.0f * fImp(ti(i-3), xi(i-3)) / (ti(i-3) - ti(i-4)) - 19.0f * fImp(ti(i-4), xi(i-4)) / dt) - 1.0f);
					} while(nn < Cuben::iterLimit && std::abs(xi(i) - ox) > Cuben::iterTol);
				} else {
					xi(i) = xi(i-1);
					dt = (1.0f/5.0f) * (ti(i) - ti(i-5));
					nn = 0;
					do {
						nn++;
						ox = xi(i);
						dx = ox - xi(i-1);
						xi(i) = ox - ((1.0f/720.0f) * dt * dt * (251.0f * fInd(ti(i), ox) / (ti(i) - ti(i-1)) + 646.0f * fInd(ti(i-1), xi(i-1)) / (ti(i-1) - ti(i-2)) - 264.0f * fInd(ti(i-2), xi(i-2)) / (ti(i-2) - ti(i-3)) + 106.0f * fInd(ti(i-3), xi(i-3)) / (ti(i-3) - ti(i-4)) - 19.0f * fInd(ti(i-4), xi(i-4)) / (ti(i-4) - ti(i-5))) - dx) / ((1.0f/720.0f) * dt * dt * (251.0f * fImp(ti(i), ox) / (ti(i) - ti(i-1)) + 646.0f * fImp(ti(i-1), xi(i-1)) / (ti(i-1) - ti(i-2)) - 264.0f * fImp(ti(i-2), xi(i-2)) / (ti(i-2) - ti(i-3)) + 106.0f * fImp(ti(i-3), xi(i-3)) / (ti(i-3) - ti(i-4)) - 19.0f * fImp(ti(i-4), xi(i-4)) / (ti(i-4) - ti(i-5))) - 1.0f);
					} while(nn < Cuben::iterLimit && std::abs(xi(i) - ox) > Cuben::iterTol);
				}
			}
			return xi;
		}
		
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

		bool test() {
			Eigen::Vector2f tInt; tInt << 0.0f,1.0f;
			Eigen::VectorXf x0(2); x0 << M_PI/2.0f,0.0f;
			Eigen::VectorXf ti; Eigen::MatrixXf xi(0,0);
			rk23Sys(pendSys, tInt, x0, ti, xi);
			std::cout << "t\tx0\tx1" << std::endl;
			for (int i = 0; i < ti.rows(); i++) {
				std::cout << ti(i) << "\t" << xi(i,0) << "\t" << xi(i,1) << std::endl;
			}
			return true;
		}
	}
}
