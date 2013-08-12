/*	Cuben::Interp static object
	Numerical interpolation
	Derived from Chapter 3 of Timothy Sauer's 'Numerical Amalysis'
*/

#include "../inc/Cuben.h"

namespace Cuben {
	namespace Interp {
		float lagrange(Eigen::VectorXf xi, Eigen::VectorXf yi, float x) {
			int n = xi.rows();
			if (n != yi.rows()) {
				throw Cuben::xMismatchedPoints();
			}
			Eigen::VectorXf t(n);
			float num, denom;
			for (int i = 0; i < n; i++) {
				num = denom = 1.0f;
				for (int j = 0; j < n; j++) {
					if (i != j) {
						num *= x - xi(j);
						denom *= xi(i) - xi(j);
					}
				}
				t(i) = yi(i) * num / denom;
			}
			return t.sum();
		}
		
		float sinInterp(float x) {
			static Cuben::Fund::Polynomial p = Cuben::Fund::Polynomial();
			if (p.getNumPoints() == 0) {
				p.push(0.0f, 0.0f);
				p.push(M_PI / 6, 0.5f);
				p.push(2 * M_PI / 6, 0.866f);
				p.push(3 * M_PI / 6, 1.0f);
			}
			float xEquiv = fmod(x, 2 * M_PI);
			if (xEquiv < 0.5 * M_PI) {
				return p.eval(xEquiv);
			} else if (xEquiv < M_PI) {
				return p.eval(M_PI - xEquiv);
			} else if (xEquiv < 1.5 * M_PI) {
				return -p.eval(xEquiv - M_PI);
			} else {
				return -p.eval(2 * M_PI - xEquiv);
			}
		}
		
		Cuben::Fund::Polynomial chebyshev(float(*f)(float), float xMin, float xMax, int n) {
			Cuben::Fund::Polynomial p;
			float x;
			for (int i = 0; i < n; i++) {
				x = 0.5 * (xMin + xMax) + 0.5 * (xMax - xMin) * std::cos((2 * (i - 1) - 1) * M_PI / (2 * n));
				p.push(x, f(x));
			}
			return p;
		}
				
		Eigen::VectorXf cheb(Eigen::VectorXf t, int order) {
			int n = t.rows();
			if (order == 0) {
				return Eigen::VectorXf::Ones(n);
			} else if (order == 1) {
				return Eigen::VectorXf(t);
			} else {
				return 2.0f * t.cwiseProduct(cheb(t, order - 1)) - cheb(t, order - 2);
			}
		}
		
		Eigen::VectorXf dchebdt(Eigen::VectorXf t, int order) {
			int n = t.rows();
			if (order == 0) {
				return Eigen::VectorXf::Zero(n);
			} else if (order == 1) {
				return Eigen::VectorXf::Ones(n);
			} else if (order == 2) {
				return 4.0f * Eigen::VectorXf(t);
			} else {
				return 2.0f * cheb(t, order - 1) + 2.0f * t.cwiseProduct(dchebdt(t, order - 1)) - dchebdt(t, order - 2);
			}
		}
		
		Eigen::VectorXf d2chebdt2(Eigen::VectorXf t, int order) {
			int n = t.rows();
			if (order == 0) {
				return Eigen::VectorXf::Zero(n);
			} else if (order == 1) {
				return Eigen::VectorXf::Zero(n);
			} else if (order == 2) {
				return 4.0f * Eigen::VectorXf::Ones(n);
			} else if (order == 3) {
				return 12.0f * t;
			} else {
				return 4.0f * dchebdt(t, order - 1) + 2 * t.cwiseProduct(d2chebdt2(t, order - 1)) - d2chebdt2(t, order - 2);
			}
		}
		
		Eigen::VectorXf chebSamp(float lhs, int n, float rhs) {
			// Return array of points sampled between [lhs,rhs] using a Chebyshev distribution
			Eigen::VectorXf xi(n);
			for (int i = 0; i < n; i++) {
				xi(i) = lhs + (rhs - lhs) * (0.5f - 0.5f * std::cos(M_PI * i / (n - 1)));
			}
			return xi;
		}

		float s(float x) {
			return std::sin(x);
		}

		CubicSplines::CubicSplines() {
			xi = Eigen::VectorXf(0);
			yi = Eigen::VectorXf(0);
			ec = EC_NATURAL;
		}
		
		void CubicSplines::push(float x, float y) {
			int n = xi.rows();
			bool isInserted = false;
			Eigen::VectorXf xiNew(n+1);
			Eigen::VectorXf yiNew(n+1);
			for (int i = 0; i <= n; i++) {
				if (isInserted) {
					xiNew(i) = xi(i-1);
					yiNew(i) = yi(i-1);
				} else if (i == n || x < xi(i)) {
					xiNew(i) = x;
					yiNew(i) = y;
					isInserted = true;
				} else {
					xiNew(i) = xi(i);
					yiNew(i) = yi(i);
				}
			}
			xi = xiNew;
			yi = yiNew;
		}

		float CubicSplines::eval(float x) {
			static Eigen::VectorXf ai(0);
			static Eigen::VectorXf bi(0);
			static Eigen::VectorXf ci(0);
			static Eigen::VectorXf di(0);
			int n = xi.rows();
			int ndxLeft = 0;

			if (x < xi(0) || xi(n-1) < x) {
				throw xOutOfInterpBounds();
			}

			if (ai.rows() != n) {
				Eigen::MatrixXf A = Eigen::MatrixXf::Zero(n,n);
				Eigen::VectorXf B = Eigen::VectorXf::Zero(n);
				float a, d, D;
				ai.resize(n);
				bi.resize(n);
				ci.resize(n);
				di.resize(n);
				
				// Apply endpoint conditions
				ai(0) = yi(0);
				ai(n-1) = yi(n-1);
				d = xi(1) - xi(0);
				D = yi(1) - yi(0);
				switch (ec) {
				case EC_CLAMPED:
					A(0,0) = 2 * d;
					A(0,1) = d;
					A(n-1,n-2) = xi(n-1) - xi(n-2);
					A(n-1,n-1) = 2 * (xi(n-1) - xi(n-2));
					B(0) = 3 * D / d;
					B(n-1) = -3 * (yi(n-1) - yi(n-2)) / (xi(n-1) - xi(n-2));
					break;
				case EC_PARABOLIC:
					A(0,0) = 1;
					A(0,1) = -1;
					A(n-1,n-2) = 1;
					A(n-1,n-1) = -1;
					B(0) = 0;
					B(n-1) = 0;
					break;
				case EC_NOTAKNOT:
					A(0,0) = xi(3) - xi(2);
					A(0,1) = -(xi(2) - xi(1) + xi(3) - xi(2));
					A(0,2) = xi(2) - xi(1);
					A(n-1,n-3) = xi(n-1) - xi(n-2);
					A(n-1,n-2) = -(xi(n-2) - xi(n-3) + xi(n-1) - xi(n-2));
					A(n-1,n-1) = xi(n-2) - xi(n-3);
					B(0) = 0;
					B(n-1) = 0;
					break;
				case EC_NATURAL:
				default:
					A(0,0) = 1.0f;
					A(n-1,n-1) = 1.0f;
					B(0) = 0;
					B(n-1) = 0;
					break;
				}

				// Loop through intermediate points
				for (int i = 1; i < n - 1; i++) {
					A(i,i-1) = d;
					A(i,i) = 2 * d + 2 * (xi(i+1) - xi(i));
					A(i,i+1) = xi(i+1) - xi(i);
					B(i) = 3 * ((yi(i+1) - yi(i)) / (xi(i+1) - xi(i)) - D / d);
					ai(i) = yi(i);
					d = xi(i+1) - xi(i);
					D = yi(i+1) - yi(i);
				}
				
				// For now, just invert to solve; then, compute other coeffs
				ci = A.inverse() * B;
				for (int i = 0; i < n - 1; i++) {
					di(i) = (ci(i+1) - ci(i)) / (3 * (xi(i+1) - xi(i)));
					bi(i) = (yi(i+1) - yi(i)) / (xi(i+1) - xi(i)) - (xi(i+1) - xi(i)) * (2 * ci(i) + ci(i+1)) / 3;
				}
			}
			while (ndxLeft < n - 1 && xi(ndxLeft+1) < x) {
				ndxLeft++;
			}
			return ai(ndxLeft) + bi(ndxLeft) * (x - xi(ndxLeft)) + ci(ndxLeft) * (x - xi(ndxLeft)) * (x - xi(ndxLeft)) + di(ndxLeft) * (x - xi(ndxLeft)) * (x - xi(ndxLeft)) * (x - xi(ndxLeft));
		}
			
		int CubicSplines::getNumPoints() {
			return xi.rows();
		}
		
		BezierSpline::BezierSpline() {
			pi = Eigen::Vector2f::Zero(2);
			pf = Eigen::Vector2f::Zero(2);
			ci = Eigen::Vector2f::Zero(2);
			cf = Eigen::Vector2f::Zero(2);
		}
		
		Eigen::Vector2f BezierSpline::eval(float t) {
			Eigen::Vector2f p;
			float bx = 3 * (ci(0) - pi(0));
			float cx = 3 * (cf(0) - ci(0)) - bx;
			float dx = pf(0) - pi(0) - bx - cx;
			float by = 3 * (ci(1) - pi(1));
			float cy = 3 * (cf(1) - ci(1)) - by;
			float dy = pf(1) - pi(1) - by - cy;
			p(0) = pi(0) + bx * t + cx * t * t + dx * t * t * t;
			p(1) = pi(1) + by * t + cy * t * t + dy * t * t * t;
			return p;
		}
		
		float cubeFit(Eigen::VectorXf xi, Eigen::VectorXf yi, float x) {
			if (xi.rows() != 4 || yi.rows() != 4) {
				throw Cuben::xMismatchedPoints();
			}
			Eigen::MatrixXf A(4,4); A << xi(0) * xi(0) * xi(0), xi(0) * xi(0), xi(0), 1.0f, xi(1) * xi(1) * xi(1), xi(1) * xi(1), xi(1), 1.0f, xi(2) * xi(2) * xi(2), xi(2) * xi(2), xi(2), 1.0f, xi(3) * xi(3) * xi(3), xi(3) * xi(3), xi(3), 1.0f;
			Eigen::VectorXf c = Cuben::Systems::paluSolve(A, yi);
			return c(0) * x * x * x + c(1) * x * x + c(2) * x + c(3);
		}

		bool test() {
			float t;
			BezierSpline bs = BezierSpline();
			bs.pi(0) = 1; bs.pi(1) = 1;
			bs.ci(0) = 1; bs.ci(1) = 3;
			bs.cf(0) = 3; bs.cf(1) = 3;
			bs.pf(0) = 2; bs.pf(1) = 2;
			t = 0.00f; std::cout << "t = " << t << "; bs(x) = " << std::endl <<  bs.eval(t) << std::endl << std::endl;
			t = 0.13f; std::cout << "t = " << t << "; bs(x) = " << std::endl <<  bs.eval(t) << std::endl << std::endl;
			t = 0.50f; std::cout << "t = " << t << "; bs(x) = " << std::endl <<  bs.eval(t) << std::endl << std::endl;
			t = 0.85f; std::cout << "t = " << t << "; bs(x) = " << std::endl <<  bs.eval(t) << std::endl << std::endl;
			t = 1.00f; std::cout << "t = " << t << "; bs(x) = " << std::endl <<  bs.eval(t) << std::endl << std::endl;
			return true;
		}
	}
}
