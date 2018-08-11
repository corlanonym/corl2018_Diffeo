//
// Created by elfuius on 12/02/18.
//

#ifndef CPP_CONVERGENCEUTILS_H
#define CPP_CONVERGENCEUTILS_H

#include "constants.h"
#include <Eigen/Dense>
#include "fileVector.h"

using namespace Eigen;
using namespace std;

template <long dimTot>
struct convPars{
public:
    convPars(const dtype alpha0, const Matrix<dtype,dimTot,1> breakRadii){_alpha0=alpha0; _breakRadii=breakRadii;};

    dtype _alpha0;
    Matrix<dtype,dimTot,1> _breakRadii;
};

template <long dimTot, long dimReal, bool doInnerContract>
void ensureRadMinConv(const Matrix<dtype, dimTot, 1> & x, Matrix<dtype, dimTot, 1> & v, const dtype alpha0, const dtype rInnerSquared[2], const dtype alphaInner){
    /**Ensure radial convergence
     * Ensure global radial convergence with the guaranteed minimal factor alpha0
     * If doInnerContract, all states within rInner are forced to converge with respect the "real" (so not internal) variables
     */

    dtype xnSquared = (dtype) x.squaredNorm();
    dtype xtv = (dtype) (x.array()*v.array()).sum();
    dtype adjustFactor;


    // Step 1 Ensure global stability
    if ((2.*xtv/xnSquared)>=alpha0){
        adjustFactor = -xtv+alpha0/2.*xnSquared;
        v += (adjustFactor/(xnSquared+dtypeEpsInv))*x;
    }


    // Step 2 Ensure inner stability if demanded
    if(doInnerContract){
        xnSquared = (dtype) (x.template block<dimReal,1>(0,0)).squaredNorm();
        if (xnSquared<rInnerSquared[1]){
            xtv = (dtype) ((x.array().template topLeftCorner<dimReal,1>())*(v.array().template topLeftCorner<dimReal,1>())).sum();
            if ((2.*xtv/xnSquared)>=alphaInner) {
                adjustFactor = -xtv + alphaInner / 2. * xnSquared;
                v.template topLeftCorner<dimReal, 1>() += (adjustFactor / (xnSquared + dtypeEpsInv)) * x.template topLeftCorner<dimReal, 1>();
            }
        }
    }
};


template <long dimTot, bool doAddConv>
void ensureRadMinConv2(const Matrix<dtype, dimTot, 1> & x, Matrix<dtype, dimTot, 1> & v, const dtype alpha0, const Matrix<dtype,dimTot,1> breakRadii){
    dtype xnSquared = (dtype) x.squaredNorm();
    dtype xtv = (dtype) (x.array()*v.array()).sum();
    dtype adjustFactor, breakDist, facBreak, vn;

    // Step 1 facilitate convergence to origin
    if (doAddConv){
        breakDist = (dtype) (x.array()/breakRadii.array()).matrix().squaredNorm();
        facBreak = exp(-breakDist);
        vn = (dtype) v.norm();

        v = (1.-facBreak)*v - facBreak*vn/(sqrt(xnSquared+dtypeEpsInv*dtypeEpsInv))*min(2.*breakDist,1.)*x;
    }

    // Step 2 Ensure global stability
    if ((2.*xtv/xnSquared)>=alpha0){
        adjustFactor = -xtv+alpha0/2.*xnSquared;
        v += (adjustFactor/(xnSquared+dtypeEpsInv))*x;
    }
}

template <long dimTot, bool doAddConv>
void ensureRadMinConv2Ptr(const dtype * const xPtr, dtype * const vPtr, const dtype alpha0, const dtype * const breakRadiiPtr ){

    //Get the map, this way data in vPtr will change too
    Map<const Array<dtype,dimTot,1>> breakRadii(breakRadiiPtr);
    Map<const Matrix<dtype,dimTot,1>> x(xPtr);
    Map<Matrix<dtype,dimTot,1>> v(vPtr);

    dtype xnSquared = (dtype) x.squaredNorm();
    dtype xtv = (dtype) (x.array()*v.array()).sum();
    dtype adjustFactor, breakDist, facBreak, vn;

    // Step 1 facilitate convergence to origin
    if (doAddConv){
        breakDist = (dtype) (x.array()/breakRadii).matrix().squaredNorm();
        facBreak = exp(-breakDist);
        vn = (dtype) v.norm();

        v = (1.-facBreak)*v - facBreak*vn/(sqrt(xnSquared+dtypeEpsInv*dtypeEpsInv))*min(2.*breakDist,1.)*x;
    }
    // Step 2 Ensure global stability
    if ((2.*xtv/xnSquared)>=alpha0){
        adjustFactor = -xtv+alpha0/2.*xnSquared;
        v += (adjustFactor/(xnSquared+dtypeEpsInv))*x;
    }
}

template <long dimTot, bool doAddConv>
class ensureRadMinConv2Class{
public:
    ensureRadMinConv2Class(){_alpha0=0.;};
    ensureRadMinConv2Class(string convergenceFile);
    ensureRadMinConv2Class(const dtype alpha0, const Matrix<dtype,dimTot,1> & breakRadiiIn);

    //~ensureRadMinConv2Class(){free(_breakRadiiArray);};

    void ensureRadMinConv2Ptr(const dtype * const xPtr, dtype * const vPtr);

    dtype _alpha0;
    //Matrix<dtype,dimTot,1> _breakRadii;
    //dtype * _breakRadiiArray;
    dtype _breakRadiiArray[dimTot];

    //EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template <long dimTot, bool doAddConv>
ensureRadMinConv2Class<dimTot, doAddConv>::ensureRadMinConv2Class(const dtype alpha0, const Matrix<dtype,dimTot,1> & breakRadiiIn){
    cout << breakRadiiIn << endl;
    _alpha0 = alpha0;
    //_breakRadiiArray = (dtype *) aligned_alloc(dimTot*sizeof(dtype), dimTot*sizeof(dtype));
    Map<Matrix<dtype,dimTot,1>> breakRadii(_breakRadiiArray);
    cout << breakRadii << endl;
    breakRadii = breakRadiiIn;
    cout << breakRadii << endl;
};

template <long dimTot, bool doAddConv>
ensureRadMinConv2Class<dimTot, doAddConv>::ensureRadMinConv2Class(string convergenceFile) {
    Matrix<dtype, dimTot+1,1> tmp = schlepil2::ReadMatrix(convergenceFile);
    cout << tmp << endl;
    //_breakRadiiArray = (dtype *) aligned_alloc(dimTot*sizeof(dtype), dimTot*sizeof(dtype));
    _alpha0 = tmp(0,0);
    Map<Matrix<dtype,dimTot,1>> breakRadii(_breakRadiiArray);
    cout << breakRadii << endl;
    breakRadii = tmp.block(1,0,dimTot,1);
    cout << breakRadii << endl;
}

template <long dimTot, bool doAddConv>
void ensureRadMinConv2Class<dimTot, doAddConv>::ensureRadMinConv2Ptr(const dtype *const xPtr, dtype *const vPtr) {
    //Get the map, this way data in vPtr will change too
    //cout << "Address of converger is " << this << endl;
    Map<const Matrix<dtype,dimTot,1>> x(xPtr);
    Map<Matrix<dtype,dimTot,1>> v(vPtr);
    Map<Matrix<dtype,dimTot,1>> breakRadii(_breakRadiiArray);
    //Matrix<dtype,dimTot,1> breakRadii;
    //for (size_t kk=0; kk<dimTot; ++kk){
    //    breakRadii(kk,0) = _breakRadiiArray[kk];
    //}

    dtype xnSquared = (dtype) x.squaredNorm();
    dtype xtv = (dtype) (x.array()*v.array()).sum();
    dtype adjustFactor, breakDist, facBreak, vn, tmpAdd;

    //cout << __LINE__ << endl;

    // Step 1 facilitate convergence to origin
    if (doAddConv){
        //cout << __LINE__ << endl;
        breakDist = (dtype) (x.array()/breakRadii.array()).matrix().squaredNorm();
        //cout << __LINE__ << endl;
        /*
        breakDist = 0.;
        cout << __LINE__ << endl;
        for (size_t kk=0; kk<dimTot; ++kk){
            cout << __LINE__ << endl;
            tmpAdd = x(kk,0);
            cout << __LINE__ << endl;
            tmpAdd /=_breakRadiiArray[kk];
            cout << __LINE__ << endl;
            tmpAdd *= tmpAdd;
            cout << __LINE__ << endl;
            breakDist += tmpAdd;
            cout << __LINE__ << endl;
        }
         */

        //cout << __LINE__ << endl;
        facBreak = exp(-breakDist);
        vn = (dtype) v.norm();

        v = (1.-facBreak)*v - facBreak*vn/(sqrt(xnSquared+dtypeEpsInv*dtypeEpsInv))*min(2.*breakDist,1.)*x;
    }
    //cout << __LINE__ << endl;
    // Step 2 Ensure global stability
    if ((2.*xtv/xnSquared)>=_alpha0){
        adjustFactor = -xtv+_alpha0/2.*xnSquared;
        v += (adjustFactor/(xnSquared+dtypeEpsInv))*x;
    }
    //cout << __LINE__ << endl;
}




#endif //CPP_CONVERGENCEUTILS_H
