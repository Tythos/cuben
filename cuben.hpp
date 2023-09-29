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
        bool isScalarWithinReltol(float actual, float expected, float relTol=1e-3);
        bool isVectorWithinReltol(Eigen::VectorXf actual, Eigen::VectorXf expected, float relTol=1e-3);
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
}