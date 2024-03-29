/**
 * equations.cpp
 */

#include "cuben.hpp"

float cuben::equations::bisect(float(*f)(float), float xLhs, float xRhs) {
    float xMid = 0.5 * (xLhs + xRhs);
    float fLhs = f(xLhs);
    float fMid = f(xMid);
    float fRhs = f(xRhs);
    if (fLhs * fRhs >= 0) {
        throw cuben::exceptions::xBisectionSign();
    }
    int k = 0;
    while (0.5 * (xRhs - xLhs) > cuben::constants::iterTol && std::abs(fMid) > cuben::constants::zeroTol && k < cuben::constants::iterLimit) {
        k++;
        if (fMid * fLhs < 0) {
            xRhs = xMid;
            fRhs = fMid;
        } else {
            xLhs = xMid;
            fLhs = fMid;
        }
        xMid = 0.5 * (xLhs + xRhs);
        fMid = f(xMid);
    }
    return xMid;    
}

float cuben::equations::fpi(float(*f)(float), float x0) {
    float xf = x0;
    int k = cuben::constants::iterLimit;
    for (int i = 0; i < k; i++) {
        xf = f(xf);
    }
    return xf;    
}

float cuben::equations::newt(float(*f)(float), float(*dfdx)(float), float x0) {
    float f0 = f(x0);
    int k = 0;
    while (std::abs(f0) > cuben::constants::zeroTol && k < cuben::constants::iterLimit) {
        x0 = x0 - f0 / dfdx(x0);
        f0 = f(x0);
        k++;
    }
    if (k == cuben::constants::iterLimit) {
        throw cuben::exceptions::xIterationLimit();
    }
    return x0;    
}

float cuben::equations::modNewt(float(*f)(float), float(*dfdx)(float), float x0, float m) {
    float f0 = f(x0);
    int k = 0;
    while (std::abs(f0) > cuben::constants::zeroTol && k < cuben::constants::iterLimit) {
        x0 = x0 - m * f0 / dfdx(x0);
        f0 = f(x0);
        k++;
    }
    if (k == cuben::constants::iterLimit) {
        throw cuben::exceptions::xIterationLimit();
    }
    return x0;
}

float cuben::equations::secant(float(*f)(float), float x0, float x1) {
    float f0 = f(x0);
    float f1 = f(x1);
    int k = 0;
    float xNext = x1;
    while (std::abs(f1) > cuben::constants::zeroTol && k < cuben::constants::iterLimit) {
        k++;
        xNext = x1 - f1 * (x1 - x0) / (f1 - f0);
        x0 = x1;
        f0 = f1;
        x1 = xNext;
        f1 = f(x1);
    }
    if (k == cuben::constants::iterLimit) {
        throw cuben::exceptions::xIterationLimit();
    }
    return x1;    
}

float cuben::equations::regulaFalsi(float(*f)(float), float xLhs, float xRhs) {
    int k = 0;
    float fLhs = f(xLhs);
    float fRhs = f(xRhs);
    float xEst = (xRhs * fLhs - xLhs * fRhs) / (fLhs - fRhs);;
    float fEst = f(xEst);
    while (std::abs(fEst) > cuben::constants::zeroTol && k < cuben::constants::iterLimit) {
        k++;
        if (fLhs * fEst < 0) {
            xRhs = xEst;
            fRhs = fEst;
        } else {
            xLhs = xEst;
            fLhs = fEst;
        }
        xEst = (xRhs * fLhs - xLhs * fRhs) / (fLhs - fRhs);
        fEst = f(xEst);
    }
    if (k == cuben::constants::iterLimit) {
        throw cuben::exceptions::xIterationLimit();
    }
    return xEst;
}

float cuben::equations::muller(float(*f)(float), float xLhs, float xRhs) {
    float xMid = 0.5 * (xLhs + xRhs);
    float fLhs = f(xLhs);
    float fMid = f(xMid);
    float fRhs = f(xRhs);
    if (fLhs * fRhs > 0) {
        throw cuben::exceptions::xBisectionSign();
    }
    int k = 0;
    Eigen::Matrix3f A;
    Eigen::Vector3f y;
    Eigen::Vector3f x;
    float det = 0.0f;
    while (std::abs(fMid) > cuben::constants::zeroTol && k < cuben::constants::iterLimit) {
        k++;
        A << xLhs*xLhs,xLhs,1, xMid*xMid,xMid,1, xRhs*xRhs,xRhs,1;
        y << fLhs, fMid, fRhs;
        x = A.colPivHouseholderQr().solve(y);
        det = x(1) * x(1) - 4.0f * x(0) * x(2);
        if (det < -cuben::constants::zeroTol) {
            throw cuben::exceptions::xComplexRoots();
        } else if (det < cuben::constants::zeroTol) {
            xMid = -x(1) / (2.0f * x(0));
        } else {
            float root1 = (-x(1) + std::sqrt(det)) / (2.0f * x(0));
            float root2 = (-x(1) - std::sqrt(det)) / (2.0f * x(0));
            if (std::abs(root1 - xMid) < std::abs(root2 - xMid)) {
                xMid = root1;
            } else {
                xMid = root2;
            }
        }
        fMid = f(xMid);
    }
    return xMid;
}

float cuben::equations::iqi(float(*f)(float), float x0, float x1) {
    float x2 = 0.5 * (x0 + x1);
    float f0 = f(x0);
    float f1 = f(x1);
    float f2 = f(x2);
    float q, r, s, x3 = 0.0f;
    if (f0 * f1 > 0) {
        throw cuben::exceptions::xBisectionSign();
    }
    int k = 0;
    while (std::abs(f2) > cuben::constants::zeroTol && k < cuben::constants::iterLimit) {
        k++;
        q = f0 / f1;
        r = f2 / f1;
        s = f2 / f0;
        x3 = x2 - (r * (r - q) * (x2 - x1) + (1 - r) * s * (x2 - x0)) / ((q - 1) * (r - 1) * (s - 1));
        x0 = x1; x1 = x2; x2 = x3;
        f0 = f1; f1 = f2; f2 = f(x3);
    }
    return x2;
}

float cuben::equations::brents(float(*f)(float), float a, float b) {
    float xi = 0.5 * (a + b);
    float fi = f(xi);
    float dx = b - a;
    float xIqi, xMuller, xBisect;
    float eIqi, eMuller, eBisect;
    float dIqi, dMuller, dBisect;
    bool isIqi, isMuller, isBisect;
    if (f(a) * f(b) > 0) {
        throw cuben::exceptions::xBisectionSign();
    }
    int k = 0;
    int realIterLimit = cuben::constants::iterLimit;
    // cuben::constants::iterLimit = 1;
    while (std::abs(fi) > cuben::constants::zeroTol && k < realIterLimit) {
        k++;
        
        // Sample solutions from IQI, Muller, and bisection
        xIqi = iqi(f, a, b);
        xMuller = muller(f, a, b);
        xBisect = bisect(f, a, b);
        
        // Measure error reduction for current and future estimates
        eIqi = f(xIqi);
        eMuller = f(xMuller);
        eBisect = f(xBisect);
        
        // Measure reduction in bracket
        dIqi = f(a) * f(xIqi) < 0 ? xIqi - a : b - xIqi;
        dMuller = f(a) * f(xMuller) < 0 ? xMuller - a : b - xMuller;
        dBisect = f(a) * f(xBisect) < 0 ? xBisect - a : b - xBisect;
        
        // Last, check to ensure solution remains bracketed
        isIqi = a < xIqi && xIqi < b;
        isMuller = a < xMuller && xMuller < b;
        isBisect = a < xBisect && xBisect < b;
        
        // Select solution and update fi, a, b
        if (eIqi < fi && dIqi <= 0.5 * (b - a) && isIqi) {
            xi = xIqi;
            fi = f(xi);
            a = f(a) * fi < 0 ? a : xi;
            b = f(a) * fi < 0 ? xi : b;
        } else if (eMuller < fi && dMuller <= 0.5 * (b - a) && isMuller) {
            xi = xMuller;
            fi = f(xi);
            a = f(a) * fi < 0 ? a : xi;
            b = f(a) * fi < 0 ? xi : b;
        } else {
            xi = xBisect;
            fi = f(xi);
            a = f(a) * fi < 0 ? a : xi;
            b = f(a) * fi < 0 ? xi : b;
        }
    }
    // cuben::constants::iterLimit = realIterLimit;
    return xi;
}
