/**
 * exceptions.hpp
*/

#pragma once 

#include <exception>

namespace cuben {
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
}
