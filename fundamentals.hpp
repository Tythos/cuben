/**
 * fundamentals.hpp
 */

#pragma once

#include <iostream>
#include <Eigen/Dense>
#include "exceptions.hpp"

namespace cuben {
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
}
