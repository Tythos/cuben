/**
 * polynomial.hpp
 */

#pragma once

#include <iostream>
#include <Eigen/Dense>

namespace cuben {
    namespace polynomial {
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
			int getNumPoints() { return ri.rows(); 	}
		};
    }
}
