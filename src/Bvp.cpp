/*	Cuben::Bvp static object
	Boundary-value problem algorithms
	Derived from Chapter 7 of Timothy Sauer's 'Numerical Amalysis'
*/

#include "../inc/Cuben.h"

namespace Cuben {
	namespace Bvp {
		void shoot(void(*dxdt)(float,Eigen::VectorXf,Eigen::VectorXf&), Eigen::Vector2f tBounds, Eigen::Vector2f xBounds, Eigen::VectorXf& ti, Eigen::MatrixXf& xi) {
			// Initial set of IVP estimates for dxdt begin with {-avg, 0, avg} ({-1,0,1} when avg < 1)
			int k = 0;
			float avg = (xBounds(1) - xBounds(0)) / (tBounds(1) - tBounds(0));
			float dxfMid = 0.0f;
			Eigen::Vector2f dxdtiOldSet;
			Eigen::Vector2f dxdtiNewSet;
			Eigen::Vector2f dxfNewSet;
			Eigen::Vector2f dxfOldSet;
			Eigen::VectorXf x0(2); x0(0) = xBounds(0);
			if (std::abs(avg) < 1.0f) {
				dxdtiOldSet << -1.0f, 1.0f;
			} else {
				dxdtiOldSet << -avg, avg;
			}
			
			// Double dxdti estimates until xf is bound by initial rates stored in dxdtiOldSet
			x0(1) = dxdtiOldSet(0); Cuben::Ode::dp45Sys(dxdt, tBounds, x0, ti, xi); dxfOldSet(0) = xi(xi.rows()-1,0) - xBounds(1);
			x0(1) = dxdtiOldSet(1); Cuben::Ode::dp45Sys(dxdt, tBounds, x0, ti, xi); dxfOldSet(1) = xi(xi.rows()-1,0) - xBounds(1);
			dxdtiNewSet = dxdtiOldSet;
			dxfNewSet = dxfOldSet;
			if (dxfOldSet(0) * dxfOldSet(1) >= 0) {
				do {
					// Rotate previous new values to current old values
					k++;
					dxdtiOldSet(0) = dxdtiNewSet(0);
					dxdtiOldSet(1) = dxdtiNewSet(1);
					dxfOldSet(0) = dxfNewSet(0);
					dxfOldSet(1) = dxfNewSet(1);
					
					// New initial rates are twice the old initial rates; propagate and store
					dxdtiNewSet(0) = 2.0f * dxdtiOldSet(0);
					dxdtiNewSet(1) = 2.0f * dxdtiOldSet(0);
					x0(1) = dxdtiNewSet(0); Cuben::Ode::dp45Sys(dxdt, tBounds, x0, ti, xi); dxfNewSet(0) = xi(xi.rows()-1,0) - xBounds(1);
					x0(1) = dxdtiNewSet(1); Cuben::Ode::dp45Sys(dxdt, tBounds, x0, ti, xi); dxfNewSet(1) = xi(xi.rows()-1,0) - xBounds(1);
				} while(dxfNewSet(0) * dxfOldSet(0) >= 0 && dxfNewSet(1) * dxfOldSet(1) >= 0 && k < Cuben::iterLimit);
				
				// Capture successful bounds in dxdtiOldSet; if not met, iteration limit was exceeded
				if (dxfNewSet(0) * dxfOldSet(0) <= 0) {
					dxdtiOldSet(1) = dxdtiOldSet(0);
					dxdtiOldSet(0) = dxdtiNewSet(0);
					dxfOldSet(1) = dxfOldSet(0);
					dxfOldSet(0) = dxfNewSet(0);
				} else if (dxfNewSet(1) * dxfOldSet(1) <= 0) {
					dxdtiOldSet(0) = dxdtiOldSet(1);
					dxdtiOldSet(1) = dxdtiNewSet(1);
					dxfOldSet(0) = dxfOldSet(1);
					dxfOldSet(1) = dxfNewSet(1);
				} else {
					throw Cuben::xIterationLimit();
				}
			}
			
			// Now we have initial rate values, stored in dxdtiOldSet, that bound the final state--bisect!
			x0(1) = 0.5f * (dxdtiOldSet(0) + dxdtiOldSet(1)); Cuben::Ode::dp45Sys(dxdt, tBounds, x0, ti, xi); dxfMid = xi(xi.rows()-1,0) - xBounds(1);
			k = 0;
			while (std::abs(dxfMid) > Cuben::bvpZeroTol && k < Cuben::iterLimit) {
				k++;
				if (dxfOldSet(0) * dxfMid <= 0) {
					dxdtiOldSet(1) = 0.5f * (dxdtiOldSet(0) + dxdtiOldSet(1));
					dxfOldSet(1) = dxfMid;
				} else if (dxfMid * dxfOldSet(1) <= 0) {
					dxdtiOldSet(0) = 0.5f * (dxdtiOldSet(0) + dxdtiOldSet(1));
					dxfOldSet(0) = dxfMid;
				} else {
					throw Cuben::xBisectionSign();
				}
				x0(1) = 0.5f * (dxdtiOldSet(0) + dxdtiOldSet(1)); Cuben::Ode::dp45Sys(dxdt, tBounds, x0, ti, xi); dxfMid = xi(xi.rows()-1,0) - xBounds(1);
			}
			
			// Having computed the appropriate boundary values, the return values ti, xi are already assigned from the most recent invocation of dp45Sys()
		}
		
		void ltiFinEl(Eigen::Vector3f coeffs, Eigen::Vector2f tBounds, Eigen::Vector2f xBounds, float dt, Eigen::VectorXf& ti, Eigen::VectorXf& xi) {
			// Solves an LTI second-order boundary value problem using the generalized finite difference method for such problems
			int n = 0;
			float lhsTerm = coeffs(0) - 0.5f * coeffs(1) * dt;
			float midTerm = -2.0f * coeffs(0) + coeffs(2) * dt * dt;
			float rhsTerm = coeffs(0) + 0.5f * coeffs(1) * dt;
			ti = Cuben::Fund::initRangeVec(tBounds(0), dt, tBounds(1)); n = ti.rows() - 2;
			Eigen::MatrixXf A(n,n);
			Eigen::VectorXf b = Eigen::VectorXf::Zero(n);

			// Compute and assign system values
			for (int i = 0; i < n; i++) {
				if (i > 0) {
					A(i,i-1) = lhsTerm;
				}
				A(i,i) = midTerm;
				if (i < n - 1) {
					A(i,i+1) = rhsTerm;
				}
			}
			b(0) = -xBounds(0) * (coeffs(0) - 0.5f * coeffs(1) * dt);
			b(n-1) = -xBounds(1) * (coeffs(0) + 0.5f * coeffs(1) * dt);
			
			// Solve for xi. which should map to ti
			std::cout << "A:" << A << std::endl;
			std::cout << "b:" << b << std::endl;
			xi = Eigen::VectorXf(n + 2);
			xi.segment(1,n) = Cuben::Systems::paluSolve(A, b);
			xi(0) = xBounds(0);
			xi(n+1) = xBounds(1);
		}
		
		void nonLinFinEl(Eigen::VectorXf(*f)(Eigen::VectorXf,float), Eigen::MatrixXf(*dfdx)(Eigen::VectorXf,float), Eigen::Vector2f tBounds, Eigen::Vector2f xBounds, float dt, Eigen::VectorXf& ti, Eigen::VectorXf& xi) {
			// Seed initial estimate as linear interpolation between bounds
			int n = 0;
			int k = 0;
			Eigen::MatrixXf jacEv;
			Eigen::VectorXf fEv;
			Eigen::MatrixXf inv;
			Eigen::VectorXf prod;
			ti = Cuben::Fund::initRangeVec(tBounds(0), dt, tBounds(1)); n = ti.rows() - 2;
			Eigen::VectorXf xWork = Eigen::VectorXf(n);
			for (int i = 0; i < n; i++) {
				xWork(i) = xBounds(0) + ti(i+1) * (xBounds(1) - xBounds(0)) / (tBounds(1) - tBounds(0));
			}
			
			// Iterate on xi using multi=variate Newton's method (modified here to include dt parameter)
			while (std::sqrt(f(xWork,dt).dot(f(xWork,dt))) > Cuben::zeroTol && k < Cuben::iterLimit) {
				xWork = xWork - dfdx(xWork,dt).inverse() * f(xWork,dt);
				k++;
			}
			if (k == Cuben::iterLimit) {
				throw Cuben::xIterationLimit();
			}
			
			// Don't forget to append boundary conditions
			xi = Eigen::VectorXf(n + 2);
			xi.segment(1,n) = xWork;
			xi(0) = xBounds(0);
			xi(n+1) = xBounds(1);
		}
		
		Eigen::VectorXf colloPoly(Eigen::Vector3f coeffs, Eigen::Vector2f tBounds, Eigen::Vector2f xBounds, float dt) {
			int n = 0;
			Eigen::MatrixXf E(0,0);
			Eigen::VectorXf g(0);
			Eigen::VectorXf ci(0);
			Eigen::VectorXf ti = Cuben::Fund::initRangeVec(tBounds(0), dt, tBounds(1));
			
			n = ti.rows();
			E = Eigen::MatrixXf(n,n);
			g = Eigen::VectorXf(n);
			
			// Construct E, g elements
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					if (std::abs(ti(i)) < Cuben::zeroTol) {
						if (j == 0) {
							E(i,j) = 1.0f;
						} else {
							E(i,j) = 0.0f;
						}
					} else if (std::abs(ti(i) - 1.0f) < Cuben::zeroTol) {
						E(i,j) = 1.0f;
					} else {
						E(i,j) = coeffs(0) * j * (j - 1) * std::pow(ti(i), j - 2) + coeffs(1) * j * std::pow(ti(i), j - 1) + coeffs(2) * std::pow(ti(i), j);
					}
				}
				if (i == 0) {
					g(i) = xBounds(0);
				} else if (i == n - 1) {
					g(i) = xBounds(1);
				} else {
					g(i) = 0.0f;
				}
			}
			
			// Solve the system to compute coefficients; then, compute yi using
			// a Van der Monde from ti and the new coefficients, ci
			return Cuben::Systems::paluSolve(E, g);
		}
		
		Eigen::VectorXf colloCheby(Eigen::Vector4f coeffs, Eigen::Vector2f tBounds, Eigen::Vector2f xBounds, int n) {
			// Initialize local variables
			float k = 2.0f / (tBounds(1) - tBounds(0));
			Eigen::VectorXf ti = Cuben::Interp::chebSamp(tBounds(0), n, tBounds(1));
			Eigen::VectorXf tti = -1.0f * Eigen::VectorXf::Ones(ti.rows()) + k * (ti - tBounds(0) * Eigen::VectorXf::Ones(ti.rows()));
			Eigen::VectorXf g;
			Eigen::MatrixXf T0, T1, T2;
			Eigen::MatrixXf E;

			// Allocate matrices based on size n (vectors will be assigned)
			T0 = Eigen::MatrixXf(n,n);
			T1 = Eigen::MatrixXf(n,n);
			T2 = Eigen::MatrixXf(n,n);
			E = Eigen::MatrixXf(n,n);
			
			// Compute Chebyshev values for times i and orders j
			for (int i = 0; i < n; i++) {
				T0.col(i) = Cuben::Interp::cheb(tti, i);
				T1.col(i) = Cuben::Interp::dchebdt(tti, i);
				T2.col(i) = Cuben::Interp::d2chebdt2(tti, i);
			}
			
			// Compute collocation matrix E
			for (int i = 0; i < n; i++) { E(0,i) = i % 2 == 0 ? 1.0f : -1.0f; }
			E.block(1,0,n-2,n) = coeffs(0) * k * k * T2.block(1,0,n-2,n) + coeffs(1) * k * T1.block(1,0,n-2,n) + coeffs(2) * T0.block(1,0,n-2,n);
			E.row(n-1) = Eigen::VectorXf::Ones(n);
			
			// Compute g and solve E c = g for coefficient vector c
			g = coeffs(3) * Eigen::VectorXf::Ones(n);
			g(0) = xBounds(0);
			g(n-1) = xBounds(1);
			return Cuben::Systems::paluSolve(E, g);
		}
		
		Eigen::VectorXf bSplineGal(Eigen::Vector3f coeffs, Eigen::Vector2f tBounds, Eigen::Vector2f xBounds, float dt) {
			int n = 0;
			Eigen::VectorXf ti = Cuben::Fund::initRangeVec(tBounds(0), dt, tBounds(1));
			Eigen::MatrixXf E = Eigen::MatrixXf::Zero(ti.rows() - 2, ti.rows() - 2);
			Eigen::VectorXf g(ti.rows() - 2);
			Eigen::VectorXf xi = Eigen::VectorXf::Zero(ti.rows());
			
			// Construct E matrix and g vector
			n = ti.rows() - 2;
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					if (j == i - 1) {
						E(i,j) = -0.5f * coeffs(0) + dt * coeffs(1) / 6.0f - 1.0f/dt;
					} else if (j == i) {
						E(i,j) = (2.0f/3.0f) * dt * coeffs(1) + 2.0f / dt;
					} else if (j == i + 1) {
						E(i,j) = 0.5f * coeffs(0) + dt * coeffs(1) / 6.0f - 1.0f/dt;
					}
				}
				
				if (i == 0) {
					g(i) = -coeffs(2) * dt - xBounds(0) * (-0.5f * coeffs(0) + dt * coeffs(1) / 6.0f - 1.0f/dt);
				} else if (i == n - 1) {
					g(i) = -coeffs(2) * dt - xBounds(1) * (0.5f * coeffs(0) + dt * coeffs(1) / 6.0f - 1.0f/dt);
				} else {
					g(i) = -coeffs(2) * dt;
				}
			}
			
			// Coefficients (and values, by virtue of b-spline collocation) are solution to Ec=g; however, this only computed the interior values; boundaries must be added before returning
			xi(0) = xBounds(0);
			xi.segment(1,n) = Cuben::Systems::paluSolve(E, g);
			xi(n+1) = xBounds(1);
			return xi;
		}
		
		void refBvp(float t, Eigen::VectorXf x, Eigen::VectorXf &dxdt) {
			dxdt = Eigen::VectorXf(2);
			dxdt(0) = x(1);
			dxdt(1) = 4.0f * x(0);
		}
		
		Eigen::VectorXf nonLinF(Eigen::VectorXf xi, float dt) {
			int n = xi.rows();
			Eigen::VectorXf fi(n);
			Eigen::Vector2f xBounds; xBounds << 1.0f,4.0f;
			for (int i = 0; i < n; i++) {
				if (i == 0) {
					fi(i) = xBounds(0) - (2.0f + dt * dt) * xi(i) + dt * dt * xi(i) * xi(i) + xi(i+1);
				} else if (i == n - 1) {
					fi(i) = xi(i-1) - (2.0f + dt * dt) * xi(i) + dt * dt * xi(i) * xi(i) + xBounds(1);
				} else {
					fi(i) = xi(i-1) - (2.0f + dt * dt) * xi(i) + dt * dt * xi(i) * xi(i) + xi(i+1);
				}
			}
			return fi;
		}
		
		Eigen::MatrixXf nonLinDfdx(Eigen::VectorXf xi, float dt) {
			int n = xi.rows();
			Eigen::MatrixXf dfdx = Eigen::MatrixXf::Zero(n,n);
			for (int i = 0; i < n; i++) {
				if (i > 0) {
					dfdx(i,i-1) = 1.0f;
				}
				if (i < n - 1) {
					dfdx(i,i+1) = 1.0f;
				}
				dfdx(i,i) = 2.0f * dt * dt * xi(i) - (2.0f + dt * dt);
			}
			return dfdx;
		}
		
		bool test() {
			Eigen::Vector3f coeffs; coeffs << 0.0f,4.0f,0.0f;
			Eigen::Vector2f tBounds; tBounds << 0.0f,1.0f;
			Eigen::Vector2f xBounds; xBounds << 1.0f,3.0f;
			Eigen::VectorXf xi;
			
			xi = bSplineGal(coeffs, tBounds, xBounds, 0.05f);
			std::cout << "b-spline coeffs/values: ";
			Cuben::Fund::printVecTrans(xi);
			std::cout << std::endl;
			return true;
		}
	}
}
