/*	Cuben::LeastSq static object
	Least squares algorithms
	Derived from Chapter 4 of Timothy Sauer's 'Numerical Amalysis'
*/

#include "cuben.h"

namespace Cuben {
	namespace LeastSq {
		Eigen::VectorXf invertNormal(Eigen::MatrixXf A, Eigen::VectorXf y) {
			if (A.rows() != y.rows()) {
				throw Cuben::xMismatchedDims();
			}
			return (A.transpose() * A).inverse() * A.transpose() * y;
		}
		
		Eigen::VectorXf fitPolynomial(Eigen::VectorXf xi, Eigen::VectorXf yi, int degree) {
			int n = xi.rows();
			Eigen::MatrixXf A(n, degree + 1);
			if (n != yi.rows()) {
				throw Cuben::xMismatchedDims();
			}
			for (int i = 0; i < n; i++) {
				A(i,0) = 1;
				for (int j = 1; j <= degree; j++) {
					A(i,j) = xi(i) * A(i,j-1);
				}
			}
			return invertNormal(A, yi);
		}
		
		Eigen::VectorXf fitPeriodic(Eigen::VectorXf xi, Eigen::VectorXf yi, int degree) {
			int n = xi.rows();
			Eigen::MatrixXf A(n, 1 + 2 * degree);
			if (n != yi.rows()) {
				throw Cuben::xMismatchedDims();
			}
			for (int i = 0; i < n; i++) {
				A(i,0) = 1;
				float freq = 1.0f;
				for (int j = 0; j < degree; j++) {
					A(i,1+2*j) = std::cos(2 * M_PI * freq * xi(i));
					A(i,2+2*j) = std::sin(2 * M_PI * freq * xi(i));
					freq = freq * 2.0f;
				}
			}
			return invertNormal(A, yi);
		}
		
		Eigen::VectorXf fitExponential(Eigen::VectorXf xi, Eigen::VectorXf yi) {
			int n = xi.rows();
			Eigen::MatrixXf A(n, 2);
			Eigen::VectorXf b(n);
			Eigen::VectorXf c(2);
			if (n != yi.rows()) {
				throw Cuben::xMismatchedDims();
			}
			for (int i = 0; i < n; i++) {
				A(i,0) = 1;
				A(i,1) = xi(i);
				b(i) = std::log(yi(i));
			}
			c = invertNormal(A, b);
			c(0) = std::exp(c(0));
			return c;
		}
		
		Eigen::VectorXf fitPower(Eigen::VectorXf xi, Eigen::VectorXf yi) {
			int n = xi.rows();
			Eigen::MatrixXf A(n, 2);
			Eigen::VectorXf b(n);
			Eigen::VectorXf c(2);
			if (n != yi.rows()) {
				throw Cuben::xMismatchedDims();
			}
			for (int i = 0; i < n; i++) {
				A(i,0) = 1;
				A(i,1) = std::log(xi(i));
				b(i) = std::log(yi(i));
			}
			c = invertNormal(A, b);
			c(0) = std::exp(c(0));
			return c;
		}

		Eigen::VectorXf fitGamma(Eigen::VectorXf xi, Eigen::VectorXf yi) {
			int n = xi.rows();
			Eigen::MatrixXf A(n, 2);
			Eigen::VectorXf b(n);
			Eigen::VectorXf c(2);
			if (n != yi.rows()) {
				throw Cuben::xMismatchedDims();
			}
			for (int i = 0; i < n; i++) {
				A(i,0) = 1;
				A(i,1) = xi(i);
				b(i) = std::log(yi(i)) - std::log(xi(i));
			}
			c = invertNormal(A, b);
			c(0) = std::exp(c(0));
			return c;
		}

		Eigen::VectorXf fitCompoundExpo(Eigen::VectorXf xi, Eigen::VectorXf yi) {
			int n = xi.rows();
			Eigen::MatrixXf A(n, 3);
			Eigen::VectorXf b(n);
			Eigen::VectorXf c(3);
			if (n != yi.rows()) {
				throw Cuben::xMismatchedDims();
			}
			for (int i = 0; i < n; i++) {
				A(i,0) = 1;
				A(i,1) = std::log(xi(i));
				A(i,2) = xi(i);
				b(i) = std::log(yi(i));
			}
			c = invertNormal(A, b);
			c(0) = std::exp(c(0));
			return c;
		}
		
		Eigen::MatrixXf computeGramSchmidt(Eigen::MatrixXf A) {
			Eigen::MatrixXf Q(A.rows(), A.cols());
			Q.col(0) = A.col(0) / A.col(0).norm();
			for (int i = 1; i < A.cols(); i++) {
				Q.col(i) = Q.col(0) * (Q.col(0).transpose() * A.col(i));
				for (int j = 1; j < i; j++) {
					Q.col(i) += Q.col(j) * (Q.col(j).transpose() * A.col(i));
				}
				Q.col(i) = A.col(i) - Q.col(i);
				Q.col(i) = Q.col(i) / Q.col(i).norm();
			}
			return Q;
		}

		void qrFactor(Eigen::MatrixXf A, Eigen::MatrixXf &Q, Eigen::MatrixXf &R) {
			Q = Eigen::MatrixXf(A.rows(), A.cols());
			R = Eigen::MatrixXf::Zero(A.cols(), A.cols());
			R(0,0) = A.col(0).norm();
			Q.col(0) = A.col(0) / R(0,0);
			for (int i = 1; i < A.cols(); i++) {
				Q.col(i) = Q.col(0) * (Q.col(0).transpose() * A.col(i));
				R(0,i) = Q.col(0).transpose() * A.col(i);
				for (int j = 1; j < i; j++) {
					R(j,i) = Q.col(j).transpose() * A.col(i);
					Q.col(i) += Q.col(j) * R(j,i);
				}
				Q.col(i) = A.col(i) - Q.col(i);
				R(i,i) = Q.col(i).norm();
				Q.col(i) = Q.col(i) / R(i,i);
			}
		}
		
		Eigen::VectorXf qrLeastSq(Eigen::MatrixXf A, Eigen::VectorXf b) {
			Eigen::MatrixXf Q(A.rows(), A.cols());
			Eigen::MatrixXf R(A.cols(), A.cols());
			qrFactor(A, Q, R);
			return R.inverse() * (Q.transpose() * b);
		}
		
		Eigen::MatrixXf householderReflector(Eigen::VectorXf a, Eigen::VectorXf b) {
			int n = a.rows();
			if (n != b.rows()) {
				throw Cuben::xMismatchedDims();
			}
			Eigen::VectorXf v = a - b;
			Eigen::MatrixXf P = v * v.transpose() / (v.transpose() * v);
			return Eigen::MatrixXf::Identity(n,n) - 2 * P;
		}
		
		void hhQrFactor(Eigen::MatrixXf A, Eigen::MatrixXf &Q, Eigen::MatrixXf &R) {
			int nIterations = std::min(A.rows() - 1, A.cols());
			Eigen::MatrixXf H(0,0);
			Eigen::VectorXf x;
			Eigen::VectorXf w;
			Eigen::MatrixXf subMat(0,0);
			Eigen::MatrixXf Hfull(0,0);

			Q = Eigen::MatrixXf::Identity(A.rows(), A.rows());
			R = A;
			for (int i = 0; i < nIterations; i++) {
				// Compute householder reflector for reduction of this column of A
				subMat = R.block(i, i, R.rows() - i, R.cols() - i);
				x = subMat.col(0);
				w = Eigen::VectorXf::Zero(x.rows());
				//w(0) = x(0) < 0 ? x.norm() : -x.norm();
				w(0) = x.norm();
				H = householderReflector(x, w);
				
				// Update Q and R computations
				Hfull = Eigen::MatrixXf::Identity(A.rows(), A.rows());
				Hfull.block(i, i, A.rows() - i, A.rows() - i) = H;
				Q = Q * Hfull;
				R = Hfull * R;
				
				// Correct R for drift
				for (int j = 0; j <= i; j++) {
					for (int k = j + 1; k < R.rows(); k++) {
						R(k,j) = 0.0f;
					}
				}
				
				// Debug report
				std::cout << "i = " << i << std::endl;
				std::cout << "subMat:" << std::endl << subMat << std::endl;
				std::cout << "H:" << std::endl << H << std::endl;
				std::cout << "Q:" << std::endl << Q << std::endl;
				std::cout << "R:" << std::endl << R << std::endl << std::endl;
			}
		}
		
		Eigen::VectorXf nonLinearGaussNewton(Eigen::VectorXf(*f)(Eigen::VectorXf), Eigen::MatrixXf(*dfdx)(Eigen::VectorXf), Eigen::VectorXf x0, int nIter) {
			Eigen::VectorXf f0 = f(x0);
			Eigen::MatrixXf dfdx0 = dfdx(x0);
			Eigen::MatrixXf A;
			Eigen::VectorXf b;
			Eigen::VectorXf dx(x0.rows()); dx(0) = Cuben::iterTol + 1.0f;
			int nf = f0.rows();
			int nx = x0.rows();
			if (nf != dfdx0.rows() || dfdx0.cols() != nx) {
				throw Cuben::xMismatchedDims();
			}
			
			int n = 0;
			while (dx.sum() > Cuben::iterTol && n < Cuben::iterLimit) {
//			for (int i = 0; i < nIter; i++) {
				n++;
				A = dfdx0.transpose() * dfdx0;
				b = -dfdx0.transpose() * f0;
				dx = A.inverse() * b;
				x0 = x0 + dx;
				f0 = f(x0);
				dfdx0 = dfdx(x0);
			}
			return x0;
		}
		
		Eigen::VectorXf fSample(Eigen::VectorXf x) {
			Eigen::VectorXf f(4);
			f(0) = std::sqrt((x(0) + 1) * (x(0) + 1) + x(1) * x(1)) - (1 + x(2));
			f(1) = std::sqrt((x(0) - 1) * (x(0) - 1) + (x(1) - 0.5) * (x(1) - 0.5)) - (0.5 + x(2));
			f(2) = std::sqrt((x(0) - 1) * (x(0) - 1) + (x(1) + 0.5) * (x(1) + 0.5)) - (0.5 + x(2));
			f(3) = std::sqrt(x(0) * x(0) + (x(1) - 1) * (x(1) - 1)) - (0.5 + x(2));
			return f;
		}
		
		Eigen::MatrixXf dfdxSample(Eigen::VectorXf x) {
			Eigen::MatrixXf dfdx(4,3);
			Eigen::VectorXf f = fSample(x);
			dfdx(0,0) = (x(0) + 1) / (f(0) + 1);
			dfdx(0,1) = x(1) / (f(0) + 1);
			dfdx(0,2) = -1;
			dfdx(1,0) = (x(0) - 1) / (f(1) + 0.5);
			dfdx(1,1) = (x(1) - 0.5) / (f(1) + 0.5);
			dfdx(1,2) = -1;
			dfdx(2,0) = (x(0) - 1) / (f(2) + 0.5);
			dfdx(2,1) = (x(1) + 0.5) / (f(2) + 0.5);
			dfdx(2,2) = -1;
			dfdx(3,0) = x(0) / (f(3) + 0.5);
			dfdx(3,1) = (x(1) - 1) / (f(3) + 0.5);
			dfdx(3,2) = -1;
			return dfdx;
		}

		bool test() {
/*			Eigen::VectorXf x(11); x << 2,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8,4.0;
			Eigen::VectorXf y(11);
			Eigen::MatrixXf A(11,8);
			for (int i = 0; i < 11; i++) {
				y(i) = 1  +  x(i)  +  x(i) * x(i)  +  x(i) * x(i) * x(i)  +  x(i) * x(i) * x(i) * x(i)  +  x(i) * x(i) * x(i) * x(i) * x(i) + x(i) * x(i) * x(i) * x(i) * x(i) * x(i) + x(i) * x(i) * x(i) * x(i) * x(i) * x(i) * x(i);
				A(i,0) = 1;
				for (int j = 1; j < 8; j++) {
					A(i,j) = A(i,j-1) * x(i);
				}
			}
			std::cout << "A: " << std::endl << A << std::endl << std::endl;
			std::cout << "y: " << std::endl << y << std::endl << std::endl;
			std::cout << "x: " << std::endl << qrLeastSq(A, y) << std::endl << std::endl;*/
			Eigen::VectorXf x = nonLinearGaussNewton(fSample, dfdxSample, Eigen::VectorXf::Zero(3), 5);
			std::cout << "x:" << std::endl << x << std::endl << std::endl;
			return true;
		}
	}
}
