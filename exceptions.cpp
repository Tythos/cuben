/**
 * exceptions.cpp
 */

#include "cuben.hpp"

const char* cuben::exceptions::xBisectionSign::what() const throw() {
    return "Signs of f(a) and f(b) must be opposed and non-zero";
}

const char* cuben::exceptions::xIterationLimit::what() const throw() {
    return "Iteration limit reached";
}

const char* cuben::exceptions::xComplexRoots::what() const throw() {
    return "Evaluation implies complex roots";
}

const char* cuben::exceptions::xZeroPivot::what() const throw() {
    return "Zero pivot encountered in matrix";
}

const char* cuben::exceptions::xMismatchedDims::what() const throw() {
    return "Matrix/vector dimensions do not match";
}

const char* cuben::exceptions::xInconvergentSystem::what() const throw() {
    return "Given system is poorly conditioned or dominant, and will not converge";
}

const char* cuben::exceptions::xMismatchedPoints::what() const throw() {
    return "Interpolation point sets must have identical dimensions";
}

const char* cuben::exceptions::xInsufficientPoints::what() const throw() {
    return "Insufficient points for interpolation";
}

const char* cuben::exceptions::xOutOfInterpBounds::what() const throw() {
    return "Given point is outside interpolation bounds";
}

const char* cuben::exceptions::xBelowMinStepSize::what() const throw() {
    return "Could not meet tolerance without reducing step size below limit";
}

const char* cuben::exceptions::xInvalidSubIndexMapping::what() const throw() {
    return "Invalid conversion between subindices and linear indices attempted; indices may be out of bounds, or dimensions may be incorrectly specified";
}

const char* cuben::exceptions::xInvalidRoll::what() const throw() {
    return "This roll cannot be performed with the current instance";
}

const char* cuben::exceptions::xInsufficientDomain::what() const throw() {
    return "The given domain has insufficient points to solve the system with this method";
}
