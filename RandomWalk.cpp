/**
 * RandomWalk.cpp
 */

#include "cuben.hpp"

cuben::RandomWalk::RandomWalk() {}

Eigen::VectorXi cuben::RandomWalk::getWalk(unsigned int nSteps) {
    Eigen::VectorXi wi = Eigen::VectorXi(nSteps);
    wi(0) = 0;
    for (int i = 1; i < nSteps; i++) {
        float seed = cuben::rand::stdRoll();
        while (seed == 0.5f) {
            seed = cuben::rand::stdRoll();
        }
        wi(i) = wi(i-1) + 2 * (seed > 0.5f) - 1;
    }
    return wi;
}
