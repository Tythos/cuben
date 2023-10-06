/**
 * cuben.hpp
 */

#pragma once

#include <exception>
#include <iostream>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#ifndef M_PI
    #define M_PI 3.1415926535897932384626433
#endif

namespace cuben {
    namespace constants {
        extern const float iterTol;
        extern const float zeroTol;
        extern const float adaptiveTol;
        extern const float relDiffEqTol;
        extern const float bvpZeroTol;
        extern const int iterLimit;
    }

    namespace exceptions {
        class xBisectionSign : public std::exception {
        public:
            virtual const char* what() const throw();
        };
        
        class xIterationLimit : public std::exception {
        public:
            virtual const char* what() const throw();
        };
        
        class xComplexRoots : public std::exception {
        public:
            virtual const char* what() const throw();
        };
        
        class xZeroPivot : public std::exception {
        public:
            virtual const char* what() const throw();
        };

        class xMismatchedDims : public std::exception {
        public:
            virtual const char* what() const throw();
        };
        
        class xInconvergentSystem : public std::exception {
        public:
            virtual const char* what() const throw();
        };
        
        class xMismatchedPoints : public std::exception {
        public:
            virtual const char* what() const throw();
        };
        
        class xInsufficientPoints : public std::exception {
        public:
            virtual const char* what() const throw();
        };
        
        class xOutOfInterpBounds : public std::exception {
        public:
            virtual const char* what() const throw();
        };
        
        class xBelowMinStepSize : public std::exception {
        public:
            virtual const char* what() const throw();
        };

        class xInvalidSubIndexMapping : public std::exception {
        public:
            virtual const char* what() const throw();
        };
        
        class xInvalidRoll : public std::exception {
        public:
            virtual const char* what() const throw();
        };
    }

    namespace fundamentals {
		void printVecTrans(Eigen::VectorXf v);
		void printVecSeries(Eigen::VectorXf v1, Eigen::VectorXf v2, char* n1="vec1", char* n2="vec2");
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
        float frobNorm(Eigen::MatrixXf M);
        bool isScalarWithinReltol(float actual, float expected, float relTol=1e-3, bool isDebuggedWhenFalse=false);
        bool isVectorWithinReltol(Eigen::VectorXf actual, Eigen::VectorXf expected, float relTol=1e-3, bool isDebuggedWhenFalse=false);
        bool isMatrixWithinReltol(Eigen::MatrixXf actual, Eigen::MatrixXf expected, float relTol=1e-3, bool isDebuggedWhenFalse=false);
    }

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
        int getNumPoints();
    };

    namespace equations {
		float bisect(float(*f)(float), float xLhs, float xRhs);
		float fpi(float(*f)(float), float x0);
		float newt(float(*f)(float), float(*dfdx)(float), float x0);
		float modNewt(float(*f)(float), float(*dfdx)(float), float x0, float m);
		float secant(float(*f)(float), float x0, float x1);
		float regulaFalsi(float(*f)(float), float xLhs, float xRhs);
		float muller(float(*f)(float), float xLhs, float xRhs);
		float iqi(float(*f)(float), float x0, float x1);
		float brents(float(*f)(float), float a, float b);
    }

    namespace systems {
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
    }

	namespace interp {
		enum EndpointCondition { EC_NATURAL, EC_CLAMPED, EC_PARABOLIC, EC_NOTAKNOT };
		float lagrange(Eigen::VectorXf xi, Eigen::VectorXf yi, float x);
		float sinInterp(float y);
		cuben::Polynomial chebyshev(float(*f)(float), float xMin, float xMax, int n);
		Eigen::VectorXf cheb(Eigen::VectorXf t, int order);
		Eigen::VectorXf dchebdt(Eigen::VectorXf t, int order);
		Eigen::VectorXf d2chebdt2(Eigen::VectorXf t, int order);
		Eigen::VectorXf chebSamp(float lhs, int n, float rhs);
		float cubeFit(Eigen::VectorXf xi, Eigen::VectorXf yi, float x);
    }

    class CubicSpline {
    private:
    protected:
        Eigen::VectorXf xi;
        Eigen::VectorXf yi;
    public:
        cuben::interp::EndpointCondition ec;
        CubicSpline();
        void push(float x, float y);
        float eval(float x);
        int getNumPoints();
    };
    
    class BezierSpline {
    private:
    protected:
        Eigen::VectorXf xi;
        Eigen::VectorXf yi;
        Eigen::VectorXf dydxi;
    public:
        BezierSpline();
        void push(float x, float y, float dydx);
        Eigen::Vector2f eval(float t);
        int getNumPoints();
    };

	namespace leastsq {
		Eigen::VectorXf invertNormal(Eigen::MatrixXf A, Eigen::VectorXf y);
        Eigen::VectorXf invertSvd(Eigen::MatrixXf A, Eigen::VectorXf y);
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
        Eigen::VectorXf nonLinearGaussNewton(Eigen::VectorXf(*f)(Eigen::VectorXf), Eigen::MatrixXf(*dfdx)(Eigen::VectorXf), Eigen::VectorXf x0, int nIter);
	}    
}
