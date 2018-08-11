//
// Created by elfuius on 12/02/18.
//

#ifndef CPP_SCHLEPILUTILS_H
#define CPP_SCHLEPILUTILS_H

#include "constants.h"
#include <iostream>

#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <string>

using namespace Eigen;
using namespace std;

void checkEndOfPath(string & path);

//Simple explicit euler
template <long dim>
void simpleIntegrate(const dtype tEnd, Matrix<dtype, dim,1> & x, function<void(const Matrix<dtype, dim,1> &, Matrix<dtype, dim,1>&)> fDyn, function<void(const Matrix<dtype, dim,1> &, Matrix<dtype, dim,1>&)> fConv, const dtype dt=5.e-4){
    assert(tEnd>=0.);
    dtype t=0.;
    dtype thisDt;
    Matrix<dtype, dim,1> xd;
    while (t<tEnd){
        thisDt = min(dt, tEnd-t);
        fDyn(x, xd);
        fConv(x, xd);
        x+= xd*thisDt;
        t+=thisDt;
    }
}

//Simple explicit euler
template <long dim>
void simpleIntegrate(const dtype tEnd, Matrix<dtype, dim,1> & x, function<void(const Matrix<dtype, dim,1> &, Matrix<dtype, dim,1>&)> & fDyn, const dtype dt=5.e-4){
    assert(tEnd>=0.);
    dtype t=0.;
    dtype thisDt;
    Matrix<dtype, dim,1> xd;
    while (t<tEnd){
        thisDt = min(dt, tEnd-t);
        fDyn(x, xd);
        x+= xd*thisDt;
        t+=thisDt;
    }
}

template <long dim>
void simpleIntegrate(const dtype tEnd, Matrix<dtype, dim,1> & x, function<void(Matrix<dtype, dim,1> &, Matrix<dtype, dim,1>&)> & fDyn, const dtype dt=5.e-4){
    assert(tEnd>=0.);
    dtype t=0.;
    dtype thisDt;
    Matrix<dtype, dim,1> xtmp, xd;
    while (t<tEnd){
        thisDt = min(dt, tEnd-t);
        xtmp = x;
        fDyn(xtmp, xd); //xtmp can be modified
        x+=xd*thisDt;
        t+=thisDt;
    }
}


template <long dim>
void simpleIntegratePtr(const dtype tEnd, dtype * const xPtr, function<void(const dtype * const, dtype * const)> fDyn, function<void(const dtype * const, dtype * const)> fConv, const dtype dtIn=1.e-3){

    if (tEnd==0.){
        return;
    }

    dtype thisDt;
    const bool dir = tEnd>=0.;
    const dtype dt = dir ? dtIn : -dtIn;
    const int nSteps = (int)floor(tEnd/dt);

    // Create velocity as array and map
    Map<Matrix<dtype, dim,1>> x(xPtr);
    dtype xdPtr[dim];
    Map<Matrix<dtype, dim,1>> xd(xdPtr);

    for (size_t k=0; k<nSteps; ++k){
        //fDyn(x, xd);
        fDyn(xPtr, xdPtr);
        //fConv(x, xd);
        fConv(xPtr, xdPtr);
        x+= xd*dt;
    }

    //Last step
    fDyn(xPtr, xdPtr);
    fConv(xPtr, xdPtr);
    x+= xd*(tEnd-((double)nSteps)*dt);
}

template <long dim>
void simpleIntegratePtr1(const dtype tEnd, dtype * const xPtr, function<void(const dtype * const, dtype * const)> fDyn, const dtype dt=5.e-4){
    assert(tEnd>=0.);
    dtype t=0.;
    dtype thisDt;

    // Create velocity as array and map
    Map<Matrix<dtype, dim,1>> x(xPtr);
    dtype xdPtr[dim];
    Map<Matrix<dtype, dim,1>> xd(xdPtr);

    while (t<tEnd){
        thisDt = min(dt, tEnd-t);
        fDyn(xPtr, xdPtr);
        x+= xd*thisDt;
        t+=thisDt;
    }
}



#endif //CPP_SCHLEPILUTILS_H
