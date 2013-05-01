/*	Cuben::Fund static object
	Basic tools (exceptions, references, etc.)
*/

#include "../inc/Cuben.h"

namespace Cuben {
	const char* xBisectionSign::what() const throw() {
		return "Signs of f(a) and f(b) must be opposed and non-zero";
	}
	
	const char* xIterationLimit::what() const throw() {
		return "Iteration limit reached";
	}
	
	const char* xComplexRoots::what() const throw() {
		return "Evaluation implies complex roots";
	}

	const char* xZeroPivot::what() const throw() {
		return "Zero pivot encountered in matrix";
	}

	const char* xMismatchedDims::what() const throw() {
		return "Matrix/vector dimensions do not match";
	}
	
	const char* xInconvergentSystem::what() const throw() {
		return "Given system is poorly conditioned or dominant, and will not converge";
	}
	
	const char* xMismatchedPoints::what() const throw() {
		return "Interpolation point sets must have identical dimensions";
	}
	
	const char* xInsufficientPoints::what() const throw() {
		return "Insufficient points for interpolation";
	}
	
	const char* xOutOfInterpBounds::what() const throw() {
		return "Given point is outside interpolation bounds";
	}

	float iterTol = 1e-8;
	float zeroTol = 1e-8;
	float adaptiveTol = 1e-4;
	int iterLimit = 1e4;
}
