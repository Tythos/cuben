/*	Cuben::Fund static object
	Fundamental numerical operations
	Derived from Chapter 0 of Timothy Sauer's 'Numerical Amalysis'
*/

#include "../inc/Cuben.h"

namespace Cuben {
	namespace Fund {
		Polynomial::Polynomial() {
			xi = Eigen::VectorXf(0);
			ci = Eigen::MatrixXf(0,0);
		}
		
		void Polynomial::print() {
			int n = this->xi.rows();
			if (n == 0) {
				std::cout << "p(x) = ?" << std::endl;
			} else {
				std::cout << "p(x) = " << this->ci(0,0);
				if (n > 1) {
					std::cout << " + ";
				}
				for (int i = 0; i < n - 1; i++) {
					if (i == n - 2) {
						std::cout << " (x - " << this->xi(i) << ") (" << this->ci(0, i + 1) << ")";
						for (int j = 0; j < n - 2; j++) {
							std::cout<< " )";
						}
					} else {
						std::cout << " (x - " << this->xi(i) << ") (" << this->ci(0, i + 1) << " +";
					}
				}
				std::cout << std::endl;
			}
		}
		
		float Polynomial::eval(float x) {
			int n = this->xi.rows();
			float y = 0.0f;
			if (n != 0) {
				for (int i = n - 1; i > 0; i--) {
					y = (x - this->xi(i - 1)) * (this->ci(0,i) + y);
				}
				y = this->ci(0,0) + y;
			}
			return y;
		}
		
		void Polynomial::push(float x, float y) {
			int n = xi.rows();
			xi.conservativeResize(n + 1);
			ci.conservativeResize(n + 1, n + 1);
			xi(n) = x;
//			std::cout << "x:" << std::endl << xi << std::endl << std::endl;
			ci(n,0) = y;
			for (int i = 1; i <= n; i++) {
//				std::cout << "ci[" << i << "]: " << std::endl << ci << std::endl << std::endl;
				ci(n-i,i) = (ci(n-i+1,i-1) - ci(n-i,i-1)) / (xi(n) - xi(n-i));
			}
		}
		
		int Polynomial::getNumPoints() {
			return xi.rows();
		}
		
		float machEps() {
			float ref = 1.0f;
			float machEps = 1.0f;
			while (ref + machEps > ref) {
				machEps = 0.5 * machEps;
			}
			return machEps;
		}
		
		Eigen::VectorXf initRangeVec(float x0, float dx, float xf) {
			int n = std::floor((xf - x0) / dx) + 1;
			Eigen::VectorXf xi(n);
			for (int i = 0; i < n; i++) {
				xi(i) = x0 + i * dx;
			}
			return xi;
		}
		
		bool test() {
			Polynomial p = Polynomial();
			p.push(0.0f,0.0f);
			p.push(M_PI/6,0.5f);
			p.push(2*M_PI/6,0.866f);
			p.push(3*M_PI/6,1.0f);
			p.print();
/*			std::cout << "p(-2.0f) = " << p.eval(-2.0f) << std::endl;
			std::cout << "p(0.0f) = " << p.eval(0.0f) << std::endl;
			std::cout << "p(1.0f) = " << p.eval(1.0f) << std::endl;
			std::cout << "p(2.0f) = " << p.eval(2.0f) << std::endl;
			std::cout << "p(4.0f) = " << p.eval(4.0f) << std::endl;*/
			return true;
		}
	}
}
