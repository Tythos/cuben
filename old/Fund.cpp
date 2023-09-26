/*	Cuben::Fund static object
	Fundamental numerical operations
	Derived from Chapter 0 of Timothy Sauer's 'Numerical Amalysis'
*/

#include "cuben.h"

namespace Cuben {
	namespace Fund {
		Polynomial::Polynomial() {
			ri = Eigen::VectorXf();
			ci = Eigen::VectorXf();
		}
		
		void Polynomial::print() {
			int n = ci.rows();
			std::cout << "P(x) = ";
			if (n == 0) {
				std::cout << "?" << std::endl;
			} else {
				for (int i = n - 1; i >= 0; i--) {
					if (i > 0) {
						std::cout << ci(i) << " + ";
					} else {
						std::cout << ci(i);
					}
					if (i > 0) {
						if (ri(i) == 0) {
							std::cout << "x * ";
						} else {
							std::cout << "(x - " << ri(i) << ") * ";
						}
						if (i > 1) std::cout << "( ";
					}
				}
				for (int i = n - 1; i > 1; i--) {
					std::cout << " )";
				}
				std::cout << std::endl;
			}
		}
		
		float Polynomial::eval(float x) {
			int n = ci.rows();
			if (n == 0) { return 0; }
			float y = ci(0);
			for (int i = 1; i < n; i++) {
				y = y * (x - ri(i)) + ci(i);
			}
			return y;
		}
		
		void Polynomial::push(float r, float c) {
			int n = ci.rows();
			ri.conservativeResize(n + 1);
			ci.conservativeResize(n + 1);
			ri(n) = r;
			ci(n) = c;
		}
		
		int Polynomial::size() {
			return ci.rows();
		}
		
		void printVecTrans(Eigen::VectorXf v) {
			std::cout << "[" << v(0);
			for (int i = 1; i < v.rows(); i++) {
				std::cout << ", " << v(i);
			}
			std::cout << "]";
		}

		void printVecSeries(Eigen::VectorXf v1, Eigen::VectorXf v2, char* n1, char* n2) {
			std::cout << n1 << "\t" << n2 << std::endl;
			std::cout << "----\t----" << std::endl;
			for (int i = 0; i < v1.rows(); i++) {
				std::cout << v1(i) << "\t" << v2(i) << std::endl;
			}
		}
		
		bool isInf(float f) {
			return std::abs(f) == std::numeric_limits<float>::infinity();
		}
		
		bool isNan(float f) {
			return !(f == f);
		}
		
		bool isFin(float f) {
			return !isInf(f) && !isNan(f);
		}
		
		float machEps() {
			float ref = 1.0f;
			float machEps = 1.0f;
			while (ref + machEps > ref) {
				machEps = 0.5 * machEps;
			}
			return machEps;
		}
		
		float relEps(float x) {
			float ref = x;
			float eps = x;
			while (ref + eps > ref) {
				eps = 0.5f * eps;
			}
			return eps;
		}
		
		float stdDev(Eigen::VectorXf xi) {
			unsigned int n = xi.rows();
			float mean = xi.sum() / (float)n;
			float run = 0.0f;
			for (int i = 0; i < n; i++) {
				run += (xi(i) - mean) * (xi(i) - mean);
			}
			return std::sqrt(run / ((float)n - 1.0f));
		}
		
		int findValue(Eigen::VectorXi vec, int value) {
			int toReturn  = -1;
			int currNdx = 0;
			while (toReturn == -1 && currNdx < vec.rows()) {
				if (vec(currNdx) == value) {
					toReturn = currNdx;
				} else {
					currNdx++;
				}
			}
			return toReturn;
		}
		
		int findValue(Eigen::VectorXf vec, float value) {
			int toReturn  = -1;
			int currNdx = 0;
			while (toReturn == -1 && currNdx < vec.rows()) {
				if (vec(currNdx) == value) {
					toReturn = currNdx;
				} else {
					currNdx++;
				}
			}
			return toReturn;
		}

		int sub2ind(Eigen::Vector2i dims, Eigen::Vector2i subNdx) {
			// Compute the linear index corresponding to the given 2d sub-indices;
			// row-major format is assumed, with initial indices at 0. For example,
			// for [3 x 5] field, the sub-indices [2,0] corresponds to linear index
			// 2, while sub-indices [0,2] corresponds to linear index 6.
			int linNdx = subNdx(0) * dims(0) + subNdx(1);
			//std::cout << "dims = [" << dims(0) << ";" << dims(1) << "], subNdx = [" << subNdx(0) << ";" << subNdx(1) << "] => linNdx = " << linNdx << std::endl;
			if (linNdx < 0 || dims(0) * dims(1) <= linNdx) {
				throw Cuben::xInvalidSubIndexMapping();
			}
			return linNdx;
		}
		
		Eigen::Vector2i ind2sub(Eigen::Vector2i dims, int linNdx) {
			// Computes the 2d coordinates (sub-indices) corresponding to the given
			// linear index as interpreted against the given table dimensions. Row-major
			// format is assumed, with initial indices at 0. FOr example, the a [3 x 5]
			// field, the linear index 2 corresponds to the sub-indices [2,0], while the
			// linear index 6 corresponds to the sub-indices [0.2].
			Eigen::Vector2i subNdcs;
			subNdcs << (int)std::floor((float)linNdx / (float)dims(0)), linNdx % dims(0);
			//std::cout << "dims = [" << dims(0) << ";" << dims(1) << "], linNdx = " << linNdx << " => subNdx = [" << subNdcs(0) << ";" << subNdcs(1) << "]" << std::endl;
			if (subNdcs(0) < 0 || dims(0) <= subNdcs(0) || subNdcs(1) < 0 || dims(1) <= subNdcs(1)) {
				throw Cuben::xInvalidSubIndexMapping();
			}
			return subNdcs;
		}
		
		Eigen::VectorXf initRangeVec(float x0, float dx, float xf) {
			float x = x0;
			int n = 0;
			while (x <= xf) {
				n++;
				x += dx;
			}
			Eigen::VectorXf xi(n);
			for (int i = 0; i < n; i++) {
				xi(i) = x0 + i * dx;
			}
			return xi;
		}
		
		Eigen::VectorXf safeResize(Eigen::VectorXf A, int nEls) {
			Eigen::VectorXf B(nEls);
			for (int i = 0; i < nEls; i++) {
				if (i < A.rows()) {
					B(i) = A(i);
				} else {
					B(i) = std::sqrt(-1);
				}
			}
			return B;
		}

		Eigen::MatrixXf safeResize(Eigen::MatrixXf A, int nRows, int nCols) {
			Eigen::MatrixXf B(nRows,nCols);
			for (int i = 0; i < nRows; i++) {
				for (int j = 0; j < nCols; j++) {
					if (i < A.rows() && j < A.cols()) {
						B(i,j) = A(i,j);
					} else {
						B(i,j) = std::sqrt(-1);
					}
				}
			}
			return B;
		}

		Eigen::MatrixXf vanDerMonde(Eigen::VectorXf x) {
			int n = x.rows();
			Eigen::MatrixXf A(n,n);
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					A(i,j) = std::pow(x(i), j);
				}
			}
			return A;
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
