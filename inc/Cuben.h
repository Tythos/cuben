/*	Cuben::Fund static object
	Basic tools (exceptions, references, etc.)
*/

#ifndef LIB_CUBEN_H
#define LIB_CUBEN_H

#include <iostream>
#include <cmath>
#include <exception>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace Cuben {
	class xBisectionSign : public std::exception {
		virtual const char* what() const throw();
	};
	
	class xIterationLimit : public std::exception {
		virtual const char* what() const throw();
	};
	
	class xComplexRoots : public std::exception {
		virtual const char* what() const throw();
	};
	
	class xZeroPivot : public std::exception {
		virtual const char* what() const throw();
	};

	class xMismatchedDims : public std::exception {
		virtual const char* what() const throw();
	};
	
	class xInconvergentSystem : public std::exception {
		virtual const char* what() const throw();
	};
	
	class xMismatchedPoints : public std::exception {
		virtual const char* what() const throw();
	};
	
	class xInsufficientPoints : public std::exception {
		virtual const char* what() const throw();
	};
	
	class xOutOfInterpBounds : public std::exception {
		virtual const char* what() const throw();
	};

	extern float iterTol;
	extern float zeroTol;
	extern float adaptiveTol;
	extern int iterLimit;

	namespace Fund {
		class Polynomial {
		private:
		protected:
			Eigen::VectorXf xi;
			Eigen::MatrixXf ci;
		public:
			Polynomial();
			void print();
			float eval(float x);
			void push(float x, float y);
			int getNumPoints();
		};
		
		float machEps();
		Eigen::VectorXf initRangeVec(float x0, float dx, float xf);
		bool test();
	}

	namespace Equations {
		float bisect(float(*f)(float), float xLhs, float xRhs);
		float fpi(float(*f)(float), float x0);
		float newt(float(*f)(float), float(*dfdx)(float), float x0);
		float modNewt(float(*f)(float), float(*dfdx)(float), float x0, float m);
		float secant(float(*f)(float), float x0, float x1);
		float regulaFalsi(float(*f)(float), float xLhs, float xRhs);
		float muller(float(*f)(float), float xLhs, float xRhs);
		float iqi(float(*f)(float), float x0, float x1);
		float brents(float(*f)(float), float a, float b);
		bool test();
	}
	
	namespace Systems {
		Eigen::VectorXf gaussElim(Eigen::MatrixXf A, Eigen::VectorXf y);
		void luFactor(Eigen::MatrixXf A, Eigen::MatrixXf &L, Eigen::MatrixXf &U);
		Eigen::VectorXf residual(Eigen::MatrixXf A, Eigen::VectorXf y, Eigen::VectorXf x);
		float relForwError(Eigen::MatrixXf A, Eigen::VectorXf y, Eigen::VectorXf xAppx, Eigen::VectorXf xExct);
		float relBackError(Eigen::MatrixXf A, Eigen::VectorXf y, Eigen::VectorXf xAppx);
		float errMagFactor(Eigen::MatrixXf A, Eigen::VectorXf y, Eigen::VectorXf xAppx, Eigen::VectorXf xExct);
		float condNum(Eigen::MatrixXf A);
		Eigen::VectorXf paluSolve(Eigen::MatrixXf A, Eigen::VectorXf y);
		void lduFactor(Eigen::MatrixXf A, Eigen::MatrixXf &L, Eigen::MatrixXf &D, Eigen::MatrixXf &U);
		bool isStrictDiagDom(Eigen::MatrixXf A);
		Eigen::VectorXf jacobiIteration(Eigen::MatrixXf A, Eigen::VectorXf y);
		Eigen::VectorXf gaussSidel(Eigen::MatrixXf A, Eigen::VectorXf y);
		Eigen::VectorXf sor(Eigen::MatrixXf A, Eigen::VectorXf y, float c);
		Eigen::VectorXf conjGrad(Eigen::MatrixXf A, Eigen::VectorXf y);
		Eigen::VectorXf multiVariateNewton(Eigen::VectorXf(*f)(Eigen::VectorXf), Eigen::MatrixXf(*dfdx)(Eigen::VectorXf), Eigen::VectorXf x0);
		Eigen::VectorXf broydenOne(Eigen::VectorXf(*f)(Eigen::VectorXf), Eigen::VectorXf x0, Eigen::VectorXf x1, Eigen::MatrixXf A0);
		Eigen::VectorXf broydenTwo(Eigen::VectorXf(*f)(Eigen::VectorXf), Eigen::VectorXf x0, Eigen::VectorXf x1, Eigen::MatrixXf B0);
		bool test();
	}

	namespace Interp {
		enum EndpointCondition { EC_NATURAL, EC_CLAMPED, EC_PARABOLIC, EC_NOTAKNOT };
		float lagrange(Eigen::VectorXf xi, Eigen::VectorXf yi, float x);
		float sinInterp(float y);
		Cuben::Fund::Polynomial chebyshev(float(*f)(float), float xMin, float xMax, int n);
		bool test();

		class CubicSplines {
		private:
		protected:
			Eigen::VectorXf xi;
			Eigen::VectorXf yi;
		public:
			EndpointCondition ec;
			CubicSplines();
			void push(float x, float y);
			float eval(float x);
			int getNumPoints();
		};
		
		class BezierSpline {
		private:
		protected:
		public:
			Eigen::Vector2f pi;
			Eigen::Vector2f pf;
			Eigen::Vector2f ci;
			Eigen::Vector2f cf;
			BezierSpline();
			Eigen::Vector2f eval(float t);
		};
	}
	
	namespace LeastSq {
		Eigen::VectorXf invertNormal(Eigen::MatrixXf A, Eigen::VectorXf y);
		Eigen::VectorXf fitPolynomial(Eigen::VectorXf xi, Eigen::VectorXf yi, int degree);
		Eigen::VectorXf fitPeriodic(Eigen::VectorXf xi, Eigen::VectorXf yi, int degree);
		Eigen::VectorXf fitExponential(Eigen::VectorXf xi, Eigen::VectorXf yi);
		Eigen::VectorXf fitPower(Eigen::VectorXf xi, Eigen::VectorXf yi);
		Eigen::VectorXf fitGamma(Eigen::VectorXf xi, Eigen::VectorXf yi);
		Eigen::VectorXf fitCompoundExpo(Eigen::VectorXf xi, Eigen::VectorXf yi);
		Eigen::MatrixXf computeGramSchmidt(Eigen::MatrixXf A);
		void qrFactor(Eigen::MatrixXf A, Eigen::MatrixXf &Q, Eigen::MatrixXf &R);
		Eigen::VectorXf qrLeastSq(Eigen::MatrixXf A, Eigen::VectorXf b);
		Eigen::MatrixXf householderReflector(Eigen::VectorXf a, Eigen::VectorXf b);
		void hhQrFactor(Eigen::MatrixXf A, Eigen::MatrixXf &Q, Eigen::MatrixXf &R);
		bool test();
	}
	
	namespace DiffInt {
		float dfdx_2pfd(float(*f)(float), float x, float h);
		float dfdx_3pcd(float(*f)(float), float x, float h);
		float d2fdx2_3pcd(float(*f)(float), float x, float h);
		float dfdx_5pcd(float(*f)(float), float x, float h);
		float d2fdx2_5pcd(float(*f)(float), float x, float h);
		float dfdx_cubic(float(*f)(float), float x, float h);
		float intfdx_trap(float(*f)(float), float x0, float x1);
		float intfdx_simp(float(*f)(float), float x0, float x2);
		float intfdx_simp38(float(*f)(float), float x0, float x3);
		float intfdx_mid(float(*f)(float), float x0, float x1);
		float intfdx_compTrap(float(*f)(float), float x0, float x1, int n);
		float intfdx_compSimp(float(*f)(float), float x0, float x1, int n);
		float intfdx_compMid(float(*f)(float), float x0, float x1, int n);
		float intfdx_romberg(float(*f)(float), float x0, float x1, int n);
		float intfdx_adaptTrap(float(*f)(float), float x0, float x1, float tol);
		float intfdx_adaptSimp(float(*f)(float), float x0, float x1, float tol);
		float intfdx_gaussQuad(float(*f)(float), float x0, float x1, int n);
		float f(float x);
		bool test();
	}
	
	namespace Ode {
		Eigen::VectorXf euler(float(*dxdt)(float, float), Eigen::VectorXf ti, float x0);
		Eigen::VectorXf trap(float(*dxdt)(float,float), Eigen::VectorXf ti, float x0);
		Eigen::VectorXf taylor2nd(float(*dxdt)(float,float), float(*d2xdt2)(float,float), float(*d2xdtdx)(float,float), Eigen::VectorXf ti, float x0);
		Eigen::VectorXf rk4(float(*dxdt)(float,float), Eigen::VectorXf ti, float x0);
		Eigen::MatrixXf eulerSys(void(*dxdt)(float,Eigen::VectorXf,Eigen::VectorXf&), Eigen::VectorXf ti, Eigen::VectorXf x0);
		Eigen::MatrixXf trapSys(void(*dxdt)(float,Eigen::VectorXf,Eigen::VectorXf&), Eigen::VectorXf ti, Eigen::VectorXf x0);
		Eigen::MatrixXf midSys(void(*dxdt)(float,Eigen::VectorXf,Eigen::VectorXf&), Eigen::VectorXf ti, Eigen::VectorXf x0);
		Eigen::MatrixXf rk4Sys(void(*dxdt)(float,Eigen::VectorXf,Eigen::VectorXf&), Eigen::VectorXf ti, Eigen::VectorXf x0);
		void dxdt(float t, Eigen::VectorXf x, Eigen::VectorXf &result);
		bool test();
	}
}

#endif
