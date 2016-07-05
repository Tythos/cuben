/*	Cuben::Fund static object
	Basic tools (exceptions, references, etc.)
*/

#ifndef LIB_CUBEN_H
#define LIB_CUBEN_H

#include <cmath>
#include <complex>
#include <cstdlib>
#include <ctime>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <exception>
#include <iostream>
#include <vector>

namespace std {
	typedef complex<float> cmplx;
}

namespace Eigen {
	typedef Matrix<std::cmplx, Dynamic, 1> VectorXc;
	typedef Matrix<std::cmplx, Dynamic, Dynamic> MatrixXc;
}

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
	
	class xBelowMinStepSize : public std::exception {
		virtual const char* what() const throw();
	};

	class xInvalidSubIndexMapping : public std::exception {
		virtual const char* what() const throw();
	};
	
	class xInvalidRoll : public std::exception {
		virtual const char* what() const throw();
	};

	extern float iterTol;
	extern float zeroTol;
	extern float adaptiveTol;
	extern float relDiffEqTol;
	extern float bvpZeroTol;
	extern int iterLimit;

	namespace Fund {
		class Polynomial {
			// Uses polynomial model given by Horner's method to minimize
			// computations: P(x) = c_n + (x - r_n) (c_{n-1} + (x - r_{n-1}...))
			// Note that this means r_0 (first base point pushed) is ignored.
			private:
			
			protected:
			Eigen::VectorXf ri;
			Eigen::VectorXf ci;
			
			public:
			Polynomial();
			void print();
			float eval(float x);
			void push(float r, float c);
			int size();
		};

		void printVecTrans(Eigen::VectorXf v);
		bool isInf(float f);
		bool isNan(float f);
		bool isFin(float f);
		float machEps();
		float relEps(float x);
		float stdDev(Eigen::VectorXf xi);
		int findValue(Eigen::VectorXi vec, int value);
		int findValue(Eigen::VectorXf vec, float value);
		int sub2ind(Eigen::Vector2i dims, Eigen::Vector2i subNdx);
		Eigen::Vector2i ind2sub(Eigen::Vector2i dims, int linNdx);
		Eigen::VectorXf initRangeVec(float x0, float dx, float xf);
		Eigen::VectorXf safeResize(Eigen::VectorXf A, int nEls);
		Eigen::MatrixXf safeResize(Eigen::MatrixXf A, int nRows, int nCols);
		Eigen::MatrixXf vanDerMonde(Eigen::VectorXf x);
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
		Eigen::VectorXf luSolve(Eigen::MatrixXf A, Eigen::VectorXf y);
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
		Eigen::VectorXf cheb(Eigen::VectorXf t, int order);
		Eigen::VectorXf dchebdt(Eigen::VectorXf t, int order);
		Eigen::VectorXf d2chebdt2(Eigen::VectorXf t, int order);
		Eigen::VectorXf chebSamp(float lhs, int n, float rhs);
		float cubeFit(Eigen::VectorXf xi, Eigen::VectorXf yi, float x);
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
		void rk23Sys(void(*dxdt)(float,Eigen::VectorXf,Eigen::VectorXf&), Eigen::Vector2f tInt, Eigen::VectorXf x0, Eigen::VectorXf& ti, Eigen::MatrixXf& xi);
		void bs23Sys(void(*dxdt)(float,Eigen::VectorXf,Eigen::VectorXf&), Eigen::Vector2f tInt, Eigen::VectorXf x0, Eigen::VectorXf& ti, Eigen::MatrixXf& xi);
		void rk45Sys(void(*dxdt)(float,Eigen::VectorXf,Eigen::VectorXf&), Eigen::Vector2f tInt, Eigen::VectorXf x0, Eigen::VectorXf& ti, Eigen::MatrixXf& xi);
		void dp45Sys(void(*dxdt)(float,Eigen::VectorXf,Eigen::VectorXf&), Eigen::Vector2f tInt, Eigen::VectorXf x0, Eigen::VectorXf& ti, Eigen::MatrixXf& xi);
		Eigen::VectorXf impEuler(float(*fInd)(float,float), float(*fImp)(float,float), Eigen::VectorXf ti, float x0);
		Eigen::VectorXf impTrap(float(*fInd)(float,float), float(*fImp)(float,float), Eigen::VectorXf ti, float x0);
		Eigen::VectorXf modab2s(float(*dxdt)(float,float), Eigen::VectorXf ti, float x0);
		Eigen::VectorXf modab3s(float(*dxdt)(float,float), Eigen::VectorXf ti, float x0);
		Eigen::VectorXf modab4s(float(*dxdt)(float,float), Eigen::VectorXf ti, float x0);
		Eigen::VectorXf modam2s(float(*fInd)(float,float), float(*fImp)(float,float), Eigen::VectorXf ti, float x0);
		Eigen::VectorXf modms2s(float(*fInd)(float,float), float(*fImp)(float,float), Eigen::VectorXf ti, float x0);
		Eigen::VectorXf modam3s(float(*fInd)(float,float), float(*fImp)(float,float), Eigen::VectorXf ti, float x0);
		Eigen::VectorXf modam4s(float(*fInd)(float,float), float(*fImp)(float,float), Eigen::VectorXf ti, float x0);
		float dxdtSng(float t, float x);
		float dxdtStiff(float t, float x);
		void pendSys(float t, Eigen::VectorXf x, Eigen::VectorXf &result);
		void scalarSys(float t, Eigen::VectorXf x, Eigen::VectorXf &result);
		bool test();
	}
	
	namespace Bvp {
		void shoot(void(*dxdt)(float,Eigen::VectorXf,Eigen::VectorXf&), Eigen::Vector2f tBounds, Eigen::Vector2f xBounds, Eigen::VectorXf& ti, Eigen::MatrixXf& xi);
		void ltiFinEl(Eigen::Vector3f coeffs, Eigen::Vector2f tBounds, Eigen::Vector2f xBounds, float dt, Eigen::VectorXf& ti, Eigen::VectorXf& xi);
		void nonLinFinEl(Eigen::VectorXf(*f)(Eigen::VectorXf,float), Eigen::MatrixXf(*dfdx)(Eigen::VectorXf,float), Eigen::Vector2f tBounds, Eigen::Vector2f xBounds, float dt, Eigen::VectorXf& ti, Eigen::VectorXf& xi);
		Eigen::VectorXf colloPoly(Eigen::Vector3f coeffs, Eigen::Vector2f tBounds, Eigen::Vector2f xBounds, float dt);
		Eigen::VectorXf colloCheby(Eigen::Vector4f coeffs, Eigen::Vector2f tBounds, Eigen::Vector2f xBounds, float dt);
		Eigen::VectorXf bSplineGal(Eigen::Vector3f coeffs, Eigen::Vector2f tBounds, Eigen::Vector2f xBounds, float dt);
		void refBvp(float t, Eigen::VectorXf x, Eigen::VectorXf &dxdt);
		Eigen::VectorXf nonLinF(Eigen::VectorXf xi, float dt);
		Eigen::MatrixXf nonLinDfdx(Eigen::VectorXf xi, float dt);
		bool test();
	}
	
	namespace Pde {
		Eigen::MatrixXf expParaEuler(Eigen::VectorXf coeffs, float(&f)(float,float,float), float(&ul)(float), float(&ur)(float), float(&u0)(float), Eigen::Vector2f xBounds, Eigen::Vector2f tBounds, float dx, float dt);
		Eigen::MatrixXf impParaEuler(Eigen::VectorXf coeffs, float(&f)(float,float), float(&ul)(float), float(&ur)(float), float(&u0)(float), Eigen::Vector2f xBounds, Eigen::Vector2f tBounds, float dx, float dt);
		Eigen::MatrixXf heatCrankNicolson(float c, float(&ul)(float), float(&ur)(float), float(&u0)(float), Eigen::Vector2f xBounds, Eigen::Vector2f tBounds, float dx, float dt);
		Eigen::MatrixXf finDiffElliptic(Eigen::VectorXf coeffs, float(&f)(float,float), float(&uxbcLower)(float), float(&uxbcUpper)(float), float(&uybcLower)(float), float(&uybcUpper)(float), Eigen::Vector2f xBounds, Eigen::Vector2f yBounds, float dx, float dy);
		float sinBound(float x);
		float zeroBound(float t);
		float expHeatForce(float u, float x, float t);
		float impHeatForce(float x, float t);
		float zeroForce(float x, float y);
		float bc1(float y);
		float bc2(float y);
		float bc3(float x);
		float bc4(float x);
		bool test();
	}
	
	namespace Rand {
		class Prng {
			protected:
				unsigned int state;
				unsigned int nRolls;
			public:
				Prng();
				Prng(unsigned int s);
				float roll();
				unsigned int getState();
		};
		
		class Lcg : public Prng {
			protected:
				unsigned int multiplier;
				unsigned int offset;
				unsigned int modulus;
			public:
				Lcg();
				Lcg(unsigned int s, unsigned int m, unsigned int o, unsigned int mod);
				float roll();
		};
		
		class MinStd : public Lcg {
			public:
				MinStd();
				MinStd(unsigned int s);
		};
		
		class Randu : public Lcg {
			public:
				Randu();
				Randu(unsigned int s);
		};
		
		class Alfg : public Prng {
			protected:
				unsigned int j;
				unsigned int k;
				Eigen::VectorXf stateVector;
				Eigen::VectorXf initialize(unsigned int nj, unsigned int nk);
			public:
				Alfg();
				Alfg(unsigned int nj, unsigned int nk);
				float roll();
		};
		
		class Mlfg : public Prng {
			protected:
				unsigned int j;
				unsigned int k;
				Eigen::VectorXf stateVector;
				Eigen::VectorXf initialize(unsigned int nj, unsigned int nk);
			public:
				Mlfg();
				Mlfg(unsigned int nj, unsigned int nk);
				float roll();
		};
		
		class MTwist : public Prng {
			protected:
				unsigned int w;
				unsigned int n;
				unsigned int m;
				unsigned int r;
				unsigned int a;
				unsigned int u;
				unsigned int s;
				unsigned int b;
				unsigned int t;
				unsigned int c;
				unsigned int l;
				unsigned int mask;
				unsigned int pow;
				unsigned int spread;
				unsigned int ndx;
				std::vector<unsigned int> stateVec;
				void generate();
			public:
				MTwist(unsigned int seed);
				float roll();
		};
		
		class Bbs : public Prng {
			protected:
				unsigned int p;
				unsigned int q;
				unsigned int xPrev;
			public:
				Bbs();
				Bbs(unsigned int np, unsigned int nq, unsigned int nx);
				float roll();
		};
		
		class Norm : public Prng {
			public:
				float mean;
				float variance;
				Norm();
				float roll();
				float cdf(float x);
		};
		
		class Halton : public Prng {
			protected:
				unsigned int basePrime;
			public:
				Halton();
				Halton(unsigned int bp);
				float roll();
				Eigen::VectorXf rollAll(unsigned int numRolls);
		};
		
		class RandomWalk {
			protected:
			public:
				RandomWalk();
				Eigen::VectorXi getWalk(unsigned int nSteps);
		};
		
		class RandomEscape : public RandomWalk {
			protected:
				unsigned int lBound;
				unsigned int uBound;
			public:
				RandomEscape();
				RandomEscape(unsigned int lb, unsigned int ub);
				Eigen::VectorXi getWalk(unsigned int nSteps = 0);
				void setBounds(unsigned lb, unsigned int ub);
		};
		
		class Brownian : public RandomWalk {
			protected:
				Norm normPrng;
			public:
				Brownian();
				Eigen::VectorXf sampleWalk(Eigen::VectorXf xi);
		};
		
		class BrownianBridge : public RandomWalk {
			protected:
			public:
				float t0;
				float x0;
				float tf;
				float xf;
				BrownianBridge();
				BrownianBridge(float nt0, float nx0, float ntf, float nxf);
				Eigen::VectorXf getWalk(float dt);
				float bbf(float,float);
				float bbg(float,float);
		};
		
		class BlackScholes : public RandomWalk {
			protected:
				float initPrice;
				float strikePrice;
				float interestRate;
				float volatility;
			public:
				BlackScholes();
				BlackScholes(float nip, float ncp, float nir, float nv);
				Eigen::VectorXf getWalk(float tf, float dt);
				float computeCallValue(float price, float tf);
		};
		
		float stdRoll();

		Eigen::VectorXf eulerMaruyama(float(*f)(float,float), float(*g)(float,float), Eigen::VectorXf ti, float x0);
		Eigen::VectorXf milstein(float(*f)(float,float), float(*g)(float,float), float(*dgdx)(float,float), Eigen::VectorXf ti, float x0);
		Eigen::VectorXf brownianBridge(Eigen::Vector2f tx0, Eigen::VectorXf txf, float resolution);

		float fTest(float,float);
		float gTest(float,float);
		float dgdxTest(float,float);
		bool test();
	}
	
	namespace Trig {
		// Raw (single-vector) transforms and their recursive components
		Eigen::VectorXc sft(Eigen::VectorXf xi);
		Eigen::VectorXf isft(Eigen::VectorXc yi);
		Eigen::VectorXc fft(Eigen::VectorXf xi);
		Eigen::MatrixXc fftRec(Eigen::VectorXf xi);
		Eigen::VectorXf ifft(Eigen::VectorXc yi);
		Eigen::MatrixXc fftRec(Eigen::VectorXc xi);
		
		// Signal transformations (includes shifting to and from symmetric format)
		Eigen::VectorXc dft(Eigen::VectorXf xi);
		Eigen::VectorXf idft(Eigen::VectorXc Xi);
		Eigen::VectorXf genFreqVec(unsigned int n, float fSamp_hz);
		Eigen::VectorXf genTimeVec(unsigned int n, float fSamp_hz);
		void dftInterp(Eigen::VectorXf tmi_s, Eigen::VectorXf xmi, unsigned int n, Eigen::VectorXf &tni_s, Eigen::VectorXf &xni);
		void dftFit(Eigen::VectorXf tni_s, Eigen::VectorXf xni, unsigned int m, Eigen::VectorXf &tmi_s, Eigen::VectorXf &xmi);
		Eigen::VectorXf wienerFilter(Eigen::VectorXf xni, float p);

		float testSignal(float t);
		bool test();
	}
	
	namespace Compress {
		Eigen::VectorXf dct(Eigen::VectorXf xi);
		float dctInterp(Eigen::VectorXf Xi, float t);
		Eigen::VectorXf dctFit(Eigen::VectorXi xi, unsigned int n);
		Eigen::MatrixXf dct2d(Eigen::MatrixXf xij);
		Eigen::MatrixXf idct2d(Eigen::MatrixXf Xij);
		Eigen::MatrixXi applyQuant(Eigen::MatrixXf Xij, Eigen::MatrixXf Qij);
		Eigen::MatrixXf reverseQuant(Eigen::MatrixXi Xij, Eigen::MatrixXf Qij);
		Eigen::MatrixXf linearQuant(unsigned int n, float p);
		Eigen::MatrixXf jpegQuant(float p);
		Eigen::MatrixXf baseYuvQuant();
		
		class JpgEncoder {
			protected:
				Eigen::VectorXi dpcmLengthTable;
				std::vector<char> dpcmHexTable;
				Eigen::MatrixXi rleLengthTable;
				std::vector<char> rleHexTable;
			public:
				JpgEncoder();
				unsigned int iitLen(int value);
				void dpcmEnc(int value, unsigned int &nBits, std::vector<char> &bits);
				void iitEnc(int value, unsigned int &nBits, std::vector<char> &bits);
				void rleEnc(unsigned int nZeros, unsigned int nLength, unsigned int &nBits, std::vector<char> &bits);
				/*std::vector<char> encode(Eigen::MatrixXi block);
				Eigen::MatrixXi decode(std::vector<char> bits);*/
		};

		Eigen::MatrixXf loadTestMatrix();
		bool test();
	}
}

#endif
