/**
 * pde.cpp
 */

#include "cuben.hpp"

Eigen::MatrixXf cuben::pde::expParaEuler(Eigen::VectorXf coeffs, float(&f)(float, float, float), float(&ul)(float), float(&ur)(float), float(&u0)(float), Eigen::Vector2f xBounds, Eigen::Vector2f tBounds, float dx, float dt) {
    Eigen::VectorXf xi = cuben::fundamentals::initRangeVec(xBounds(0), dx, xBounds(1));
    Eigen::VectorXf ti = cuben::fundamentals::initRangeVec(tBounds(0), dt, tBounds(1));
    int nx = xi.rows();
    int nt = ti.rows();
    Eigen::MatrixXf uij(nx,nt);
    Eigen::MatrixXf G = Eigen::MatrixXf::Zero(nx,nx);
    Eigen::MatrixXf H = Eigen::MatrixXf::Zero(nx,nx);
    Eigen::VectorXf fv = Eigen::VectorXf::Zero(nx);
    Eigen::VectorXf bv = Eigen::VectorXf::Zero(nx);
    float omega;
    float alpha, beta, gamma;
    float delta, epsilon, zeta;
    
    // Assign boundary conditions to t0, xl, and xr blocks of uij
    for (int i = 0; i < nx; i++) {
        uij(i,0) = u0(xi(i));
    }
    for (int j = 0; j < nt; j++) {
        uij(0,j) = ul(ti(j));
        uij(nx-1,j) = ur(ti(j));
    }
    
    // First-order ut equations
    omega = -(dt * dt) / (coeffs(2) + dt * coeffs(4));
    beta = -2.0f * coeffs(0) / (dx * dx) - 2.0f * coeffs(2) / (dt * dt) - coeffs(4) / dt;
    epsilon = coeffs(2) / (dt * dt);
    
    // Second-order ut equations
//			omega = -(dt * dt) / (Ceoffs(2) + 0.5f * dt * coeffs(4));
//			beta = -2.0f * coeffs(0) / (dx * dx) - 2.0f * coeffs(2) / (dt * dt);
//			epsilonm = coeffs(2) / (dt * dt);
    
    // With uniform step sizes, we can compute G and H matrices only once, as they
    // will be constant for all time points
    alpha = coeffs(0) / (dx * dx) - 0.5f * coeffs(1) / (dx * dt) - 0.5f * coeffs(3) / dx;
    gamma = coeffs(0) / (dx * dx) + 0.5f * coeffs(1) / (dx * dt) + 0.5f * coeffs(3) / dx;
    delta = 0.5f * coeffs(1) / (dx * dt);
    zeta = -0.5f * coeffs(1) / (dx * dt);
    for (int i = 0; i < nx; i++) {
        if (i > 0) {
            G(i,i-1) = alpha;
            H(i,i-1) = delta;
        }
        G(i,i) = beta;
        H(i,i) = epsilon;
        if (i < nx - 1) {
            G(i,i+1) = gamma;
            H(i,i+1) = zeta;
        }
    }
    
    // Beginning at the second timestep, update the computation according to the
    // equation u*j = omega * (G u*j-1 + H u*j-2 + fv + bv)
    int j2ndx;
    for (int j = 1; j < nt; j++) {
        if (j == 1) {
            j2ndx = 0;
        } else {
            j2ndx = j - 2;
        }
        for (int i = 0; i < nx; i++) {
            fv(i) = f(uij(i,j-1), xi(i), ti(j-1));
        }
        bv(0) = alpha * ul(ti(j-1)) + delta * ul(ti(j2ndx));
        bv(nx-1) = gamma * ur(ti(j-1)) + zeta * ur(ti(j2ndx));
        uij.col(j) = omega * (G * uij.col(j-1) + H * uij.col(j2ndx) + fv + bv);
    }
    return uij;
}

Eigen::MatrixXf cuben::pde::impParaEuler(Eigen::VectorXf coeffs, float(&f)(float, float), float(&ul)(float), float(&ur)(float), float(&u0)(float), Eigen::Vector2f xBounds, Eigen::Vector2f tBounds, float dx, float dt) {
    Eigen::VectorXf xi = cuben::fundamentals::initRangeVec(xBounds(0), dx, xBounds(1));
    Eigen::VectorXf ti = cuben::fundamentals::initRangeVec(tBounds(0), dt, tBounds(1));
    int nx = xi.rows();
    int nt = ti.rows();
    Eigen::MatrixXf uij(nx,nt);
    Eigen::MatrixXf G = Eigen::MatrixXf::Zero(nx,nx);
    Eigen::MatrixXf H = Eigen::MatrixXf::Zero(nx,nx);
    Eigen::MatrixXf invNegG(nx,nx);
    Eigen::VectorXf fv = Eigen::VectorXf::Zero(nx);
    Eigen::VectorXf bv = Eigen::VectorXf::Zero(nx);
    float omega;
    float alpha, beta, gamma;
    float delta, epsilon, zeta;
    
    // Assign boundary conditions to t0, xl, and xr blocks of uij
    for (int i = 0; i < nx; i++) {
        uij(i,0) = u0(xi(i));
    }
    for (int j = 0; j < nt; j++) {
        uij(0,j) = ul(ti(j));
        uij(nx-1,j) = ur(ti(j));
    }
    
    // With uniform step sizes, we can compute G and H matrices only once, as they
    // will be constant for all time points
    alpha = coeffs(0) / (dx * dx) - 0.5f * coeffs(1) / (dx * dt) - 0.5f * coeffs(3) / dx;
    beta = -2.0f * coeffs(0) / (dx * dx) + coeffs(2) / (dt * dt) + coeffs(4) / dt + coeffs(5);
    gamma = coeffs(0) / (dx * dx) + 0.5f * coeffs(1) / (dx * dt) + 0.5f * coeffs(3) / dx;
    delta = 0.5f * coeffs(1) / (dx * dt);
    epsilon = -2.0f * coeffs(2) / (dt * dt) - coeffs(4) / dt;
    zeta = -0.5f * coeffs(1) / (dx * dt);
    for (int i = 0; i < nx; i++) {
        if (i > 0) {
            G(i,i-1) = alpha;
            H(i,i-1) = delta;
        }
        G(i,i) = beta;
        H(i,i) = epsilon;
        if (i < nx - 1) {
            G(i,i+1) = gamma;
            H(i,i+1) = zeta;
        }
    }
    
    // Invert G once to reuse in each iteration
    invNegG = (-G).inverse();
    
    // Beginning at the second timestep, update the computation according to the
    // equation u*j = omega * (G u*j-1 + H u*j-2 + fv + bv)
    int j2ndx;
    for (int j = 1; j < nt; j++) {
        if (j == 1) {
            j2ndx = 0;
        } else {
            j2ndx = j - 2;
        }
        for (int i = 0; i < nx; i++) {
            fv(i) = f(xi(i), ti(j));
        }
        bv(0) = alpha * ul(ti(j)) + delta * ul(ti(j2ndx));
        bv(nx-1) = gamma * ur(ti(j)) + zeta * ur(ti(j2ndx));
        uij.col(j) = invNegG * (H * uij.col(j-1) + coeffs(2) * uij.col(j2ndx) / (dt * dt) + fv + bv);
    }
    return uij;
}

Eigen::MatrixXf cuben::pde::heatCrankNicolson(float c, float(&ul)(float), float(&ur)(float), float(&u0)(float), Eigen::Vector2f xBounds, Eigen::Vector2f tBounds, float dx, float dt) {
    // Note that Crank-Nicolson is inapporpriate for the generalized parabolic PDE, so only the heat-specific algorithm is implemented here
    float sigma = c * dt / (dx * dx);
    Eigen::VectorXf xi = cuben::fundamentals::initRangeVec(xBounds(0), dx, xBounds(1));
    Eigen::VectorXf ti = cuben::fundamentals::initRangeVec(tBounds(0), dt, tBounds(1));
    int nx = xi.rows();
    int nt = ti.rows();
    Eigen::MatrixXf A = Eigen::MatrixXf::Zero(nx-2,nx-2);
    Eigen::MatrixXf B = Eigen::MatrixXf::Zero(nx-2,nx-2);
    Eigen::MatrixXf Ainv(nx-2,nx-2);
    Eigen::MatrixXf uij(nx,nt);
    Eigen::VectorXf bv = Eigen::VectorXf::Zero(nx-2);
    
    // Assign boundary conditions to uij
    for (int i = 0; i < nx; i++) {
        uij(i,0) = u0(xi(i));
    }
    for (int j = 0; j < nt; j++) {
        uij(0,j) = ul(ti(j));
        uij(nx-1,j) = ur(ti(j));
    }
    
    // Compure constant values for each row of A, B, and inv(A)
    for (int i = 0; i < nx - 2; i++) {
        if (i > 0) {
            A(i,i-1) = -sigma;
            B(i,i-1) = sigma;
        }
        A(i,i) = 2.0f + 2.0f * sigma;
        B(i,i) = 2.0f - 2.0f * sigma;
        if (i < nx - 3) {
            A(i,i+1) = -sigma;
            B(i,i+1) = sigma;
        }
    }
    Ainv = A.inverse();
    
    // March through each timestep to compute the intermediate values of uij
    for (int j = 1; j < nt; j++) {
        bv(0) = uij(0,j-1) + uij(0,j);
        bv(nx-3) = uij(nx-1,j-1) + uij(nx-1,j);
        //bv(nx-3) = uij(nx-1,j-1) + uij(nx-1,j-1);
        uij.col(j).segment(1,nx-2) = Ainv * (B * uij.col(j-1).segment(1,nx-2) + sigma * bv);
    }
    return uij;
}

Eigen::MatrixXf cuben::pde::finDiffElliptic(Eigen::VectorXf coeffs, float(&f)(float,float), float(&uxbcLower)(float), float(&uxbcUpper)(float), float(&uybcLower)(float), float(&uybcUpper)(float), Eigen::Vector2f xBounds, Eigen::Vector2f yBounds, float dx, float dy) {
    Eigen::VectorXf xi = cuben::fundamentals::initRangeVec(xBounds(0), dx, xBounds(1));
    Eigen::VectorXf yi = cuben::fundamentals::initRangeVec(yBounds(0), dy, yBounds(1));
    int nx = xi.rows();
    int ny = yi.rows();
    int nInt = (nx - 2) * (ny - 2);
    Eigen::MatrixXf M = Eigen::MatrixXf::Zero(nInt,nInt);
    Eigen::VectorXf b = Eigen::VectorXf::Zero(nInt);
    Eigen::Vector2i intDims; intDims << ny-2,nx-2;
    bool isBoundaryFilter[9];
    
    // Initialize state matrix and assign boundary values. Average BCs at corners.
    Eigen::MatrixXf uij(nx,ny);
    for (int j = 1; j < ny-1; j++) {
        uij(0,j) = uxbcLower(yi(j));
        uij(nx-1,j) = uxbcUpper(yi(j));
    }
    for (int i = 1; i < nx-1; i++) {
        uij(i,0) = uybcLower(xi(i));
        uij(i,ny-1) = uybcUpper(xi(i));
    }
    uij(0,0) = 0.5f * (uxbcLower(yi(0)) + uybcLower(xi(0)));
    uij(0,ny-1) = 0.5f * (uxbcLower(yi(ny-1)) + uybcUpper(xi(0)));
    uij(nx-1,0) = 0.5f * (uxbcUpper(yi(0)) + uybcLower(xi(nx-1)));
    uij(nx-1,ny-1) = 0.5f * (uxbcUpper(yi(ny-1)) + uybcUpper(xi(nx-1)));
    
    // Compute matrix xoefficients, assembled into an array for better indexing
    Eigen::VectorXf matCoeffs(9);
    matCoeffs(0) = 0.25 * coeffs(1) / (dx * dy);
    matCoeffs(1) = coeffs(2) / (dy * dy) - 0.5f * coeffs(4) / dy;
    matCoeffs(2) = -0.25f * coeffs(1) / (dx * dy);
    matCoeffs(3) = coeffs(0) / (dx * dx) - 0.5f * coeffs(3) / dx;
    matCoeffs(4) = -2.0f * coeffs(0) / (dx * dx) - 2.0f * coeffs(2) / (dy * dy) + coeffs(5);
    matCoeffs(5) = coeffs(0) / (dx * dx) + 0.5f * coeffs(3) / dx;
    matCoeffs(6) = -0.25f * coeffs(1) / (dx * dy);
    matCoeffs(7) = coeffs(2) / (dy * dy) + 0.25f * coeffs(4) / dy;
    matCoeffs(8) = 0.25f * coeffs(1) / (dx * dy);
    //std::cout << "matCoeffs: " << matCoeffs << std::endl;
    
    // Iterate through all elements, evaluating forcing term and assigning each coefficient to M or b
    // Note that i.j are indices for the net matrix, including boundary points. Relative indices iJndcs,jNdcs
    // are defined within the context of the interior point matrix. Note that the 2d/linear index conversion
    // must take place in [y,x] and not [x,y] to ensure linear indexing is consistent with usage here ([0,0],
    // [1,0], [2,0], etc.)
    for (int i = 1; i < nx - 1; i++) {
        for (int j = 1; j < ny - 1; j++) {
            // Examine location to construct boundary filter
            isBoundaryFilter[0] = i == 1 || j == 1;
            isBoundaryFilter[1] = j == 1;
            isBoundaryFilter[2] = i == nx - 2 || j == 1;
            isBoundaryFilter[3] = i == 1;
            isBoundaryFilter[4] = false;
            isBoundaryFilter[5] = i == nx - 2;
            isBoundaryFilter[6] = i == 1 || j == ny - 2;
            isBoundaryFilter[7] = j == ny - 2;
            isBoundaryFilter[8] = i == nx - 2 || j == ny - 2;
            
            // Construct linear indices for each relative coordinate not on the boundary
            Eigen::VectorXi iNdcs(9); iNdcs << i-2, i-1, i, i-2, i-1, i, i-2, i-1, i;
            Eigen::VectorXi jNdcs(9); jNdcs << j-2, j-2, j-2, j-1, j-1, j-1, j, j, j;
            Eigen::VectorXi linNdcs(9);
            for (int k = 0; k < 9; k++) {
                //std::cout << "[i,j,k] = [" << i << "," << j << "," << k << "]" << std::endl;
                if (!isBoundaryFilter[k]) {
                    Eigen::Vector2i subNdx; subNdx << jNdcs(k), iNdcs(k);
                    linNdcs(k) = cuben::fundamentals::sub2ind(intDims, subNdx);
                }
            }
            
            // Compute forcing term and boundary values from stored uij
            b(linNdcs(4)) = -f(xi(i), yi(i));
            Eigen::VectorXf bv(9);
            bv << uij(i-1,j-1), uij(i,j-1), uij(i+1,j-1), uij(i-1,j), uij(i,j), uij(i+1,j), uij(i-1,j+1), uij(i,j+1), uij(i+1,j+1);
            
            // Filter coefficients to one side or another; if filter value is true, coefficient will
            // be subtracted from RHS instead of added to indexed location on LHS matrix
            for (int k = 0; k < 9; k++) {
                if (isBoundaryFilter[k]) {
                    b(linNdcs(4)) = b(linNdcs(4)) - matCoeffs(k) * bv(k);
                } else {
                    M(linNdcs(4),linNdcs(k)) = matCoeffs(k);
                }
            }
        }
    }
    
    // Solve system and file elements into uij
    //std::cout << "M: " << M << std::endl;
    //std::cout << "b: " << b << std::endl;
    Eigen::VectorXf uVec = cuben::systems::paluSolve(M,b);
    for (int i = 1; i < nx - 1; i++) {
        for (int j = 1; j < ny - 1; j++) {
            Eigen::Vector2i subNdx; subNdx << j-1,i-1;
            int linNdx = cuben::fundamentals::sub2ind(intDims, subNdx);
            uij(i,j) = uVec(linNdx);
        }
    }
    return uij;
}
