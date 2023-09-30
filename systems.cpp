/**
 * systems.cpp
 */

#include "cuben.hpp"

Eigen::VectorXf cuben::systems::gaussElim(Eigen::MatrixXf A, Eigen::VectorXf y) {
    int n = y.size();
    Eigen::VectorXf x(n);
    float ratio = 0.0f;
    if (n != A.rows()) {
        throw cuben::exceptions::xMismatchedDims();
    }
    
    // Reduce to triangular form
    for (int j = 0; j < n; j++) {
        if (std::abs(A(j,j)) < cuben::fundamentals::machEps()) {
            throw cuben::exceptions::xZeroPivot();
        }
        for (int i = j + 1; i < n; i++) {
            ratio = A(i,j) / A(j,j);
            for (int k = j + 1; k < n; k++) {
                A(i,k) = A(i,k) - ratio * A(j,k);
            }
            y(i) = y(i) - ratio * y(j);
        }
    }
    
    // Back-substitute to solve for x
    for (int i = n - 1; i >= 0; i--) {
        for (int j = i + 1; j < n; j++) {
            y(i) = y(i) - A(i,j) * x(j);
        }
        x(i) = y(i) / A(i,i);
    }
return x;
}

void cuben::systems::luFactor(Eigen::MatrixXf A, Eigen::MatrixXf &L, Eigen::MatrixXf &U) {
    int n = A.rows();
    if (n != L.rows() || n != U.rows()) {
        throw cuben::exceptions::xMismatchedDims();
    }
    L = Eigen::MatrixXf::Identity(n,n);
    U = Eigen::MatrixXf(A);
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            L(i,j) = U(i,j) / U(j,j);
            U.row(i) = U.row(i) - L(i,j) * U.row(j);
        }
    }
}

Eigen::VectorXf cuben::systems::luSolve(Eigen::MatrixXf A, Eigen::VectorXf y) {
    int n = A.rows();
    if (n != y.rows()) {
        throw cuben::exceptions::xMismatchedDims();
    }
    Eigen::MatrixXf L(n,n);
    Eigen::MatrixXf U(n,n);
    luFactor(A, L, U);

    // Solve Lc = b for c
    Eigen::VectorXf c = Eigen::VectorXf::Zero(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            c(i) = c(i) + L(i,j) * c(j);
        }
        c(i) = y(i) - c(i);
    }
    
    // Solve Ux = c for x
    Eigen::VectorXf x = Eigen::VectorXf::Zero(n);
    for (int i = n - 1; i >= 0; i--) {
        for (int j = n - 1; j > i; j--) {
            x(i) = x(i) + U(i,j) * x(j);
        }
        x(i) = (c(i) - x(i)) / U(i,i);
    }
    return x;
}

Eigen::VectorXf cuben::systems::residual(Eigen::MatrixXf A, Eigen::VectorXf y, Eigen::VectorXf x) {
    int n = A.rows();
    if (n != y.rows() || n != x.rows()) {
        throw cuben::exceptions::xMismatchedDims();
    }
    return y - A * x;
}

float cuben::systems::relForwError(Eigen::MatrixXf A, Eigen::VectorXf y, Eigen::VectorXf xAppx, Eigen::VectorXf xExct) {
    int n = A.rows();
    if (n != y.rows() || n != xAppx.rows() || n != xExct.rows()) {
        throw cuben::exceptions::xMismatchedDims();
    }
    return std::sqrt((xExct - xAppx).dot(xExct - xAppx)) / std::sqrt(xExct.dot(xExct));
}

float cuben::systems::relBackError(Eigen::MatrixXf A, Eigen::VectorXf y, Eigen::VectorXf xAppx) {
    Eigen::VectorXf r = residual(A, y, xAppx);
    return std::sqrt(r.dot(r)) / std::sqrt(y.dot(y));
}

float cuben::systems::errMagFactor(Eigen::MatrixXf A, Eigen::VectorXf y, Eigen::VectorXf xAppx, Eigen::VectorXf xExct) {
    return relForwError(A, y, xAppx, xExct) / relBackError(A, y, xAppx);
}

float cuben::systems::condNum(Eigen::MatrixXf A) {
    int n = A.rows();
    Eigen::MatrixXf I = A.inverse();
    float aMaxCond = 0.0f;
    float iMaxCond = 0.0f;
    float aThisCond, iThisCond;
    for (int i = 0; i < n; i++) {
        aThisCond = 0.0f;
        iThisCond = 0.0f;
        for (int j = 0; j < n; j++) {
            aThisCond += std::abs(A(i,j));
            iThisCond += std::abs(I(i,j));
        }
        if (aThisCond > aMaxCond) { aMaxCond = aThisCond; }
        if (iThisCond > iMaxCond) { iMaxCond = iThisCond; }
    }
    return aMaxCond * iMaxCond;
}

Eigen::VectorXf cuben::systems::paluSolve(Eigen::MatrixXf A, Eigen::VectorXf y) {
    // Rearrange A to ensure the largest sequential values are along
    // each diagonal, then invoke luSolve() to solve
    int n = A.rows();
    int ndx = 0;
    Eigen::VectorXf swap(n);
    float v;
    if (n != y.rows()) {
        throw cuben::exceptions::xMismatchedDims();
    }
    for (int i = 0; i < n; i++) {
        // Swap the ith row with the row whose ith column has the
        // biggest value, not including those already swapped
        ndx = i;
        for (int j = i + 1; j < n; j++) {
            if (A(j,i) > A(ndx,i)) {
                ndx = j;
            }
        }
        if (ndx != i) {
            swap = A.row(ndx); A.row(ndx) = A.row(i); A.row(i) = swap;
            v = y(ndx); y(ndx) = y(i); y(i) = v;
        }
    }
    return luSolve(A, y);
}

void cuben::systems::lduFactor(Eigen::MatrixXf A, Eigen::MatrixXf &L, Eigen::MatrixXf &D, Eigen::MatrixXf &U) {
    int n = A.rows();
    if (n != L.rows() || n != D.rows() || n != U.rows()) {
        throw cuben::exceptions::xMismatchedDims();
    }
    L = Eigen::MatrixXf::Zero(n,n);
    D = Eigen::MatrixXf::Zero(n,n);
    U = Eigen::MatrixXf::Zero(n,n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i < j) {
                L(i,j) = A(i,j);
            } else if (i == j) {
                D(i,j) = A(i,j);
            } else {
                U(i,j) = A(i,j);
            }
        }
    }
}

bool cuben::systems::isStrictDiagDom(Eigen::MatrixXf A) {
    bool toReturn = true;
    int n = A.rows();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (A(i,j) > A(i,i)) { toReturn = false; }
        }
    }
    return toReturn;
}

Eigen::VectorXf cuben::systems::jacobiIteration(Eigen::MatrixXf A, Eigen::VectorXf y) {
    int n = A.rows();
    if (n != y.rows()) {
        throw cuben::exceptions::xMismatchedDims();
    }
    if (!isStrictDiagDom(A)) {
        throw cuben::exceptions::xInconvergentSystem();
    }
    Eigen::VectorXf x(y);
    Eigen::MatrixXf L(n,n);
    Eigen::MatrixXf D(n,n);
    Eigen::MatrixXf U(n,n);
    Eigen::MatrixXf Dinv(n,n);
    lduFactor(A, L, D, U);
    Dinv = D.inverse();
    int k = 0;
    while (relBackError(A, y, x) > cuben::constants::iterTol && k < cuben::constants::iterLimit) {
        x = Dinv * (y - (L + U) * x);
        k++;
    }
    return x;
}

Eigen::VectorXf cuben::systems::gaussSidel(Eigen::MatrixXf A, Eigen::VectorXf y) {
    int n = A.rows();
    if (n != y.rows()) {
        throw cuben::exceptions::xMismatchedDims();
    }
    if (!isStrictDiagDom(A)) {
        throw cuben::exceptions::xInconvergentSystem();
    }
    Eigen::VectorXf x(y);
    Eigen::MatrixXf L(n,n);
    Eigen::MatrixXf D(n,n);
    Eigen::MatrixXf U(n,n);
    Eigen::MatrixXf Dinv(n,n);
    Eigen::MatrixXf DlhsInv(n,n);
    lduFactor(A, L, D, U);
    Dinv = D.inverse();
    DlhsInv = (Dinv * L + Eigen::MatrixXf::Identity(n,n)).inverse();
    int k = 0;
    while (relBackError(A, y, x) > cuben::constants::iterTol && k < cuben::constants::iterLimit) {
        x = DlhsInv * Dinv * (y - U * x);
        k++;
    }
    return x;
}

Eigen::VectorXf cuben::systems::sor(Eigen::MatrixXf A, Eigen::VectorXf y, float c) {
    int n = A.rows();
    if (n != y.rows()) {
        throw cuben::exceptions::xMismatchedDims();
    }
    if (!isStrictDiagDom(A)) {
        throw cuben::exceptions::xInconvergentSystem();
    }
    Eigen::VectorXf x(y);
    Eigen::MatrixXf L(n,n);
    Eigen::MatrixXf D(n,n);
    Eigen::MatrixXf U(n,n);
    Eigen::MatrixXf cLDinv(n,n);
    Eigen::MatrixXf DcLinv(n,n);
    lduFactor(A, L, D, U);
    cLDinv = (c * L + D).inverse();
    DcLinv = (D + c * L).inverse();
    int k = 0;
    while (relBackError(A, y, x) > cuben::constants::iterTol && k < cuben::constants::iterLimit) {
        //x = (1 - c) * x + c * DlhsInv * Dinv * (y - U * x);
        x = cLDinv * ((1 - c) * D * x - c * U * x) + c * DcLinv * y;
        k++;
    }
    return x;
}

Eigen::VectorXf cuben::systems::conjGrad(Eigen::MatrixXf A, Eigen::VectorXf y) {
    // Assumes A is symmetric positive definite; adding a check for this
    // would be a good idea, if one can be found or derived
    int n = A.rows();
    if (n != y.rows()) {
        throw cuben::exceptions::xMismatchedDims();
    }
    Eigen::VectorXf x = Eigen::VectorXf::Zero(n);
    Eigen::VectorXf d = y;
    Eigen::VectorXf r = y;
    Eigen::VectorXf rNew = Eigen::VectorXf::Zero(n);
    float alph = 0.0f;
    float beta = 0.0f;
    int k = 0;
    while (std::abs(r.dot(r)) > cuben::constants::zeroTol && k < cuben::constants::iterLimit) {
        k++;
        alph = r.dot(r) / (d.transpose() * A * d);
        x = x + alph * d;
        rNew = r - alph * A * d;
        beta = rNew.dot(rNew) / r.dot(r);
        d = rNew + beta * d;
        r = rNew;
    }
    return x;
}

Eigen::VectorXf cuben::systems::multiVariateNewton(Eigen::VectorXf(*f)(Eigen::VectorXf), Eigen::MatrixXf(*dfdx)(Eigen::VectorXf), Eigen::VectorXf x0) {
    int n = x0.rows();
    if (n != f(x0).rows() || n != dfdx(x0).rows()) {
        throw cuben::exceptions::xMismatchedDims();
    }
    int k = 0;
    while (std::sqrt(f(x0).dot(f(x0))) > cuben::constants::zeroTol && k < cuben::constants::iterLimit) {
        x0 = x0 - dfdx(x0).inverse() * f(x0);
        k++;
    }
    return x0;
}

Eigen::VectorXf cuben::systems::broydenOne(Eigen::VectorXf(*f)(Eigen::VectorXf), Eigen::VectorXf x0, Eigen::VectorXf x1, Eigen::MatrixXf A0) {
    int n = x0.rows();
    if (n != f(x0).rows() || n != x1.rows() || n != A0.rows()) {
        throw cuben::exceptions::xMismatchedDims();
    }
    Eigen::VectorXf d(n);
    Eigen::VectorXf D(n);
    int k = 0;
    while (std::sqrt(f(x1).dot(f(x1))) > cuben::constants::zeroTol && k < cuben::constants::iterLimit) {
        k++;
        d = x1 - x0;
        D = f(x1) - f(x0);
        A0 = A0 + (D - A0 * d) * d.transpose() / d.dot(d);
        x0 = x1;
        x1 = x1 - A0.inverse() * f(x1);
    }
    return x1;
}

Eigen::VectorXf cuben::systems::broydenTwo(Eigen::VectorXf(*f)(Eigen::VectorXf), Eigen::VectorXf x0, Eigen::VectorXf x1, Eigen::MatrixXf B0) {
    int n = x0.rows();
    if (n != f(x0).rows() || n != x1.rows() || n != B0.rows()) {
        throw cuben::exceptions::xMismatchedDims();
    }
    Eigen::VectorXf d(n);
    Eigen::VectorXf D(n);
    int k = 0;
    while (std::sqrt(f(x1).dot(f(x1))) > cuben::constants::zeroTol && k < cuben::constants::iterLimit) {
        k++;
        d = x1 - x0;
        D = f(x1) - f(x0);
        B0 = B0 + ((d - B0 * D) * d.transpose() * B0) / (d.transpose() * B0 * D);
        x0 = x1;
        x1 = x1 - B0 * f(x1);
    }
    return x1;
}
