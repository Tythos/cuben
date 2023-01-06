/*	Cuben::Systems static object
	Systems of equations
	Derived from Chapter 2 of Timothy Sauer's 'NUmerical Amalysis'
*/

#include "cuben.h"

namespace Cuben {
	namespace Systems {
		Eigen::VectorXf gaussElim(Eigen::MatrixXf A, Eigen::VectorXf y) {
			int n = y.size();
			Eigen::VectorXf x(n);
			float ratio = 0.0f;
			if (n != A.rows()) {
				throw Cuben::xMismatchedDims();
			}
			
			// Reduce to triangular form
			for (int j = 0; j < n; j++) {
				if (std::abs(A(j,j)) < Cuben::Fund::machEps()) {
					throw Cuben::xZeroPivot();
				}
				for (int i = j + 1; i < n; i++) {
					ratio = A(i,j) / A(j,j);
					for (int k = j + 1; k < n; k++) {
						A(i,k) = A(i,k) - ratio * A(j,k);
					}
					y(i) = y(i) - ratio * y(j);
				}
			}
			
			// Back-substitute to solve for x
			for (int i = n - 1; i >= 0; i--) {
				for (int j = i + 1; j < n; j++) {
					y(i) = y(i) - A(i,j) * x(j);
				}
				x(i) = y(i) / A(i,i);
			}
			return x;
		}
		
		void luFactor(Eigen::MatrixXf A, Eigen::MatrixXf &L, Eigen::MatrixXf &U) {
			int n = A.rows();
			if (n != L.rows() || n != U.rows()) {
				throw Cuben::xMismatchedDims();
			}
			L = Eigen::MatrixXf::Identity(n,n);
			U = Eigen::MatrixXf(A);
			for (int i = 1; i < n; i++) {
				for (int j = 0; j < i; j++) {
					L(i,j) = U(i,j) / U(j,j);
					U.row(i) = U.row(i) - L(i,j) * U.row(j);
				}
			}
		}
		
		Eigen::VectorXf luSolve(Eigen::MatrixXf A, Eigen::VectorXf y) {
			int n = A.rows();
			if (n != y.rows()) {
				throw Cuben::xMismatchedDims();
			}
			Eigen::MatrixXf L(n,n);
			Eigen::MatrixXf U(n,n);
			luFactor(A, L, U);

			// Solve Lc = b for c
			Eigen::VectorXf c = Eigen::VectorXf::Zero(n);
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < i; j++) {
					c(i) = c(i) + L(i,j) * c(j);
				}
				c(i) = y(i) - c(i);
			}
			
			// Solve Ux = c for x
			Eigen::VectorXf x = Eigen::VectorXf::Zero(n);
			for (int i = n - 1; i >= 0; i--) {
				for (int j = n - 1; j > i; j--) {
					x(i) = x(i) + U(i,j) * x(j);
				}
				x(i) = (c(i) - x(i)) / U(i,i);
			}
			return x;
		}
		
		Eigen::VectorXf residual(Eigen::MatrixXf A, Eigen::VectorXf y, Eigen::VectorXf x) {
			int n = A.rows();
			if (n != y.rows() || n != x.rows()) {
				throw Cuben::xMismatchedDims();
			}
			return y - A * x;
		}
		
		float relForwError(Eigen::MatrixXf A, Eigen::VectorXf y, Eigen::VectorXf xAppx, Eigen::VectorXf xExct) {
			int n = A.rows();
			if (n != y.rows() || n != xAppx.rows() || n != xExct.rows()) {
				throw Cuben::xMismatchedDims();
			}
			return std::sqrt((xExct - xAppx).dot(xExct - xAppx)) / std::sqrt(xExct.dot(xExct));
		}
		
		float relBackError(Eigen::MatrixXf A, Eigen::VectorXf y, Eigen::VectorXf xAppx) {
			Eigen::VectorXf r = residual(A, y, xAppx);
			return std::sqrt(r.dot(r)) / std::sqrt(y.dot(y));
		}
		
		float errMagFactor(Eigen::MatrixXf A, Eigen::VectorXf y, Eigen::VectorXf xAppx, Eigen::VectorXf xExct) {
			return relForwError(A, y, xAppx, xExct) / relBackError(A, y, xAppx);
		}
		
		float condNum(Eigen::MatrixXf A) {
			int n = A.rows();
			Eigen::MatrixXf I = A.inverse();
			float aMaxCond = 0.0f;
			float iMaxCond = 0.0f;
			float aThisCond, iThisCond;
			for (int i = 0; i < n; i++) {
				aThisCond = 0.0f;
				iThisCond = 0.0f;
				for (int j = 0; j < n; j++) {
					aThisCond += std::abs(A(i,j));
					iThisCond += std::abs(I(i,j));
				}
				if (aThisCond > aMaxCond) { aMaxCond = aThisCond; }
				if (iThisCond > iMaxCond) { iMaxCond = iThisCond; }
			}
			return aMaxCond * iMaxCond;
		}
		
		Eigen::VectorXf paluSolve(Eigen::MatrixXf A, Eigen::VectorXf y) {
			// Rearrange A to ensure the largest sequential values are along
			// each diagonal, then invoke luSolve() to solve
			int n = A.rows();
			int ndx = 0;
			Eigen::VectorXf swap(n);
			float v;
			if (n != y.rows()) {
				throw Cuben::xMismatchedDims();
			}
			for (int i = 0; i < n; i++) {
				// Swap the ith row with the row whose ith column has the
				// biggest value, not including those already swapped
				ndx = i;
				for (int j = i + 1; j < n; j++) {
					if (A(j,i) > A(ndx,i)) {
						ndx = j;
					}
				}
				if (ndx != i) {
					swap = A.row(ndx); A.row(ndx) = A.row(i); A.row(i) = swap;
					v = y(ndx); y(ndx) = y(i); y(i) = v;
				}
			}
			return luSolve(A, y);
		}
		
		void lduFactor(Eigen::MatrixXf A, Eigen::MatrixXf &L, Eigen::MatrixXf &D, Eigen::MatrixXf &U) {
			int n = A.rows();
			if (n != L.rows() || n != D.rows() || n != U.rows()) {
				throw Cuben::xMismatchedDims();
			}
			L = Eigen::MatrixXf::Zero(n,n);
			D = Eigen::MatrixXf::Zero(n,n);
			U = Eigen::MatrixXf::Zero(n,n);
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					if (i < j) {
						L(i,j) = A(i,j);
					} else if (i == j) {
						D(i,j) = A(i,j);
					} else {
						U(i,j) = A(i,j);
					}
				}
			}
		}
		
		bool isStrictDiagDom(Eigen::MatrixXf A) {
			bool toReturn = true;
			int n = A.rows();
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					if (A(i,j) > A(i,i)) { toReturn = false; }
				}
			}
			return toReturn;
		}
		
		Eigen::VectorXf jacobiIteration(Eigen::MatrixXf A, Eigen::VectorXf y) {
			int n = A.rows();
			if (n != y.rows()) {
				throw Cuben::xMismatchedDims();
			}
			if (!isStrictDiagDom(A)) {
				throw Cuben::xInconvergentSystem();
			}
			Eigen::VectorXf x(y);
			Eigen::MatrixXf L(n,n);
			Eigen::MatrixXf D(n,n);
			Eigen::MatrixXf U(n,n);
			Eigen::MatrixXf Dinv(n,n);
			lduFactor(A, L, D, U);
			Dinv = D.inverse();
			int k = 0;
			while (relBackError(A, y, x) > Cuben::iterTol && k < Cuben::iterLimit) {
				x = Dinv * (y - (L + U) * x);
				k++;
			}
			return x;
		}
		
		Eigen::VectorXf gaussSidel(Eigen::MatrixXf A, Eigen::VectorXf y) {
			int n = A.rows();
			if (n != y.rows()) {
				throw Cuben::xMismatchedDims();
			}
			if (!isStrictDiagDom(A)) {
				throw Cuben::xInconvergentSystem();
			}
			Eigen::VectorXf x(y);
			Eigen::MatrixXf L(n,n);
			Eigen::MatrixXf D(n,n);
			Eigen::MatrixXf U(n,n);
			Eigen::MatrixXf Dinv(n,n);
			Eigen::MatrixXf DlhsInv(n,n);
			lduFactor(A, L, D, U);
			Dinv = D.inverse();
			DlhsInv = (Dinv * L + Eigen::MatrixXf::Identity(n,n)).inverse();
			int k = 0;
			while (relBackError(A, y, x) > Cuben::iterTol && k < Cuben::iterLimit) {
				x = DlhsInv * Dinv * (y - U * x);
				k++;
			}
			return x;
		}
		
		Eigen::VectorXf sor(Eigen::MatrixXf A, Eigen::VectorXf y, float c) {
			int n = A.rows();
			if (n != y.rows()) {
				throw Cuben::xMismatchedDims();
			}
			if (!isStrictDiagDom(A)) {
				throw Cuben::xInconvergentSystem();
			}
			Eigen::VectorXf x(y);
			Eigen::MatrixXf L(n,n);
			Eigen::MatrixXf D(n,n);
			Eigen::MatrixXf U(n,n);
			Eigen::MatrixXf cLDinv(n,n);
			Eigen::MatrixXf DcLinv(n,n);
			lduFactor(A, L, D, U);
			cLDinv = (c * L + D).inverse();
			DcLinv = (D + c * L).inverse();
			int k = 0;
			while (relBackError(A, y, x) > Cuben::iterTol && k < Cuben::iterLimit) {
				//x = (1 - c) * x + c * DlhsInv * Dinv * (y - U * x);
				x = cLDinv * ((1 - c) * D * x - c * U * x) + c * DcLinv * y;
				k++;
			}
			return x;
		}
		
		Eigen::VectorXf conjGrad(Eigen::MatrixXf A, Eigen::VectorXf y) {
			// Assumes A is symmetric positive definite; adding a check for this
			// would be a good idea, if one can be found or derived
			int n = A.rows();
			if (n != y.rows()) {
				throw Cuben::xMismatchedDims();
			}
			Eigen::VectorXf x = Eigen::VectorXf::Zero(n);
			Eigen::VectorXf d = y;
			Eigen::VectorXf r = y;
			Eigen::VectorXf rNew = Eigen::VectorXf::Zero(n);
			float alph = 0.0f;
			float beta = 0.0f;
			int k = 0;
			while (std::abs(r.dot(r)) > Cuben::zeroTol && k < Cuben::iterLimit) {
				k++;
				alph = r.dot(r) / (d.transpose() * A * d);
				x = x + alph * d;
				rNew = r - alph * A * d;
				beta = rNew.dot(rNew) / r.dot(r);
				d = rNew + beta * d;
				r = rNew;
			}
			return x;
		}
		
		Eigen::VectorXf multiVariateNewton(Eigen::VectorXf(*f)(Eigen::VectorXf), Eigen::MatrixXf(*dfdx)(Eigen::VectorXf), Eigen::VectorXf x0) {
			int n = x0.rows();
			if (n != f(x0).rows() || n != dfdx(x0).rows()) {
				throw Cuben::xMismatchedDims();
			}
			int k = 0;
			while (std::sqrt(f(x0).dot(f(x0))) > Cuben::zeroTol && k < Cuben::iterLimit) {
				x0 = x0 - dfdx(x0).inverse() * f(x0);
				k++;
			}
			return x0;
		}
		
		Eigen::VectorXf broydenOne(Eigen::VectorXf(*f)(Eigen::VectorXf), Eigen::VectorXf x0, Eigen::VectorXf x1, Eigen::MatrixXf A0) {
			int n = x0.rows();
			if (n != f(x0).rows() || n != x1.rows() || n != A0.rows()) {
				throw Cuben::xMismatchedDims();
			}
			Eigen::VectorXf d(n);
			Eigen::VectorXf D(n);
			int k = 0;
			while (std::sqrt(f(x1).dot(f(x1))) > Cuben::zeroTol && k < Cuben::iterLimit) {
				k++;
				d = x1 - x0;
				D = f(x1) - f(x0);
				A0 = A0 + (D - A0 * d) * d.transpose() / d.dot(d);
				x0 = x1;
				x1 = x1 - A0.inverse() * f(x1);
			}
			return x1;
		}
		
		Eigen::VectorXf broydenTwo(Eigen::VectorXf(*f)(Eigen::VectorXf), Eigen::VectorXf x0, Eigen::VectorXf x1, Eigen::MatrixXf B0) {
			int n = x0.rows();
			if (n != f(x0).rows() || n != x1.rows() || n != B0.rows()) {
				throw Cuben::xMismatchedDims();
			}
			Eigen::VectorXf d(n);
			Eigen::VectorXf D(n);
			int k = 0;
			while (std::sqrt(f(x1).dot(f(x1))) > Cuben::zeroTol && k < Cuben::iterLimit) {
				k++;
				d = x1 - x0;
				D = f(x1) - f(x0);
				B0 = B0 + ((d - B0 * D) * d.transpose() * B0) / (d.transpose() * B0 * D);
				x0 = x1;
				x1 = x1 - B0 * f(x1);
			}
			return x1;
		}
		
		Eigen::VectorXf testf(Eigen::VectorXf x) {
			Eigen::VectorXf f(2);
			float u = x(0); float v = x(1);
			f(0) = 6.0f * u * u * u + u * v - 3.0f * v * v * v - 4.0f;
			f(1) = u * u - 18.0f * u * v * v + 16.0f * v * v * v + 1.0f;
			return f;
		}
		
		Eigen::MatrixXf testdfdx(Eigen::VectorXf x) {
			Eigen::MatrixXf dfdx(2,2);
			float u = x(0); float v = x(1);
			dfdx(0,0) = 18.0f * u * u + v;
			dfdx(0,1) = u - 9.0f * v * v;
			dfdx(1,0) = 2.0f * u - 18.0f * v * v;
			dfdx(1,1) = -36.0f * u * v + 48.0f * v * v;
			return dfdx;
		}

		bool test() {
			std::cout.precision(16);
			Eigen::VectorXf x0(2); x0 << 1.00f,-0.50f;
			Eigen::VectorXf x1(2); x0 << 0.90f,-0.35f;
			for (int i = 0; i < 16; i++) {
				Cuben::iterLimit = i + 1;
				std::cout << "x (" << i + 1 << " iterations): " << broydenTwo(&testf, x0, x1, Eigen::MatrixXf::Identity(2,2)).transpose() << std::endl;
			}
			return true;
		}
	}
}
