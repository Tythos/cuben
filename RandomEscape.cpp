/**
 * RandomEscape.cpp
 */

#include "cuben.hpp"

cuben::RandomEscape::RandomEscape() {
    lBound = -1;
    uBound = 1;
}

cuben::RandomEscape::RandomEscape(signed int lb, signed int ub) {
    lBound = lb;
    uBound = ub;
}

Eigen::VectorXi cuben::RandomEscape::getWalk(unsigned int nSteps) {
    nSteps = 1024;
    RandomWalk rw = RandomWalk();
    Eigen::VectorXi wi = rw.getWalk(nSteps);
    
    while (wi.minCoeff() > -(int)lBound && wi.maxCoeff() < (int)uBound) {
        nSteps = 2 * nSteps;
        wi = rw.getWalk(nSteps);
    }
    
    int lNdx = cuben::fundamentals::findValue(wi, -(int)lBound);
    int uNdx = cuben::fundamentals::findValue(wi, (int)uBound);
    if (lNdx < uNdx && lNdx > -1 || uNdx < 0) {
        wi = wi.segment(0, lNdx + 1);
    } else {
        wi = wi.segment(0, uNdx + 1);
    }
    return wi;
}

void cuben::RandomEscape::setBounds(signed lb, signed int ub) {
    lBound = lb;
    uBound = ub;
}

const Eigen::Vector2i cuben::RandomEscape::getBounds() {
    Eigen::Vector2i bounds; bounds <<
        lBound, uBound;
    return bounds;
}
