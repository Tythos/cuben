/*	Cuben::Fund static object
	Basic tools (exceptions, references, etc.)
*/

#include "cuben.h"

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

	const char* xBelowMinStepSize::what() const throw() {
		return "Could not meet tolerance without reducing step size below limit";
	}
	
	const char * xInvalidSubIndexMapping::what() const throw() {
		return "Invalid conversion between subindices and linear indices attempted; indices may be out of bounds, or dimensions may be incorrectly specified";
	}
	
	const char * xInvalidRoll::what() const throw() {
		return "This roll cannot be performed with the current instance";
	}
	
	float iterTol = 1e-8;
	float zeroTol = 1e-8;
	float adaptiveTol = 1e-4;
	float relDiffEqTol = 1e-6;
	float bvpZeroTol = 1e-6;
	int iterLimit = 1e4;
}
