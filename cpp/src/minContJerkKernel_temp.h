//
// Created by elfuius on 08/02/18.
//

#ifndef CPP_KERNEL_H
#define CPP_KERNEL_H

#include "constants.h"
#include "polyEval.h"
#include <Eigen/Dense>
#include <string>
#include <iostream>

using namespace Eigen;
using namespace std;

template <long dim>
class kernel{
public:
    kernel(Matrix<dtype, dim,1> center, Matrix<dtype, dim, 1> trans, dtype b, dtype e);
    kernel(const string defString);

    static long nPars(){return dim+dim+3;};

    inline Matrix<dtype, dim,1> getCenter()const{return _center;};
    inline const Matrix<dtype, dim,1> & getCenterR()const{return _center;};

    inline Matrix<dtype, dim,1> getTrans()const{return _trans;};
    inline const Matrix<dtype, dim,1> & getTransR()const{return _trans;};

    dtype outOfBaseCoef()const{return _e;};
    dtype minCoef()const{return _e;};
    dtype maxCoef()const{return (dtype) 1.;};


    inline dtype kerVal(const dtype xIn);
    inline dtype kerVal(const Matrix<dtype, dim,1> & xIn);
    inline dtype kerValD(const dtype xIn);
    inline dtype kerValD(const Matrix<dtype, dim,1> & xIn);
    inline dtype kerValDS(const dtype xIn);
    inline dtype kerValDS(const Matrix<dtype, dim,1> & xIn);
    inline void kerValnD(const dtype xIn, dtype (&valND)[2]);
    inline void kerValnD(const Matrix<dtype, dim,1> & xIn, dtype (&valND)[2]);
    inline void kerValnDS(const dtype xIn, dtype (&valND)[2]);
    inline void kerValnDS(const Matrix<dtype, dim,1> & xIn, dtype (&valND)[2]);

    inline void kerValnDeriv(const Matrix<dtype, dim,1> & xIn, dtype (&valND)[2]);

//private:
public: // Debug

    void computeInternal();

    Matrix<dtype, dim,1> _center;
    Matrix<dtype, dim,1> _trans;
    dtype _b,_e;
    dtype _coeffs[6];

    static const int _nOrd=5;

    Matrix<dtype, dim,1> _dxTemp;

};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<long dim>
kernel<dim>::kernel(Matrix<dtype, dim, 1> center, Matrix<dtype, dim, 1> trans, dtype b, dtype e) {
    _center = center;
    _trans = trans;
    _b = b;
    _e = e;

    computeInternal();
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<long dim>
kernel<dim>::kernel(const string defString) {
    //Construct the istringstream
    istringstream defStringStream(defString);
    string line;

    dtype tmpVal;
    long tmpDim;

    //Expected format

    // dimenstion
    // center
    // translation
    // base
    // evalue

    defStringStream >> tmpDim;
    assert( (tmpDim==dim) && "Incompatible dimension definition");

    //Start with center
    for (size_t i=0; i<dim; ++i){
        defStringStream >> tmpVal;
        _center(i) = tmpVal;
    }
    //Continue with translation
    for (size_t i=0; i<dim; ++i){
        defStringStream >> tmpVal;
        _trans(i) = tmpVal;
    }
    //Rest
    defStringStream >> tmpVal;
    _b = tmpVal;
    defStringStream >> tmpVal;
    _e = tmpVal;
    computeInternal();
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<long dim>
void kernel<dim>::computeInternal(){

    _coeffs[0] = (dtype) __c0_cpp;
    _coeffs[1] = (dtype) __c1_cpp;
    _coeffs[2] = (dtype) __c2_cpp;
    _coeffs[3] = (dtype) __c3_cpp;
    _coeffs[4] = (dtype) __c4_cpp;
    _coeffs[5] = (dtype) __c5_cpp;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<long dim>
inline dtype kernel<dim>::kerVal(const dtype xIn){
    //Check which case
    if(xIn>_b){
        // Constant influence outside base
        return _e;
    }else{
        // fifth order poly
        return polyEval(xIn, _coeffs, _nOrd);
    }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<long dim>
inline dtype kernel<dim>::kerVal(const Matrix<dtype, dim, 1> &xIn) {
    return kerVal( (dtype) (xIn-_center).norm() );
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<long dim>
inline dtype kernel<dim>::kerValD(const dtype xIn){
    //Check which case
    if(xIn>_b){
        // Constant influence outside base
        return 0.;
    }else{
        // derivative of fifth order poly
        return polyEvalDeriv(xIn, _coeffs, _nOrd);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<long dim>
inline dtype kernel<dim>::kerValD(const Matrix<dtype, dim, 1> &xIn) {
    return kerValD( (dtype) (xIn-_center).norm() );
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<long dim>
inline dtype kernel<dim>::kerValDS(const dtype xIn) {
    return kerValD( xIn )/(xIn+dtype_eps);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<long dim>
inline dtype kernel<dim>::kerValDS(const Matrix<dtype, dim, 1> &xIn) {
    return kerValDS( (dtype) (xIn-_center).norm() );
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<long dim>
inline void kernel<dim>::kerValnD(const dtype xIn, dtype (&valND)[2]) {
    valND[0] = kerVal(xIn);
    valND[1] = kerValD(xIn);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<long dim>
inline void kernel<dim>::kerValnD(const Matrix<dtype, dim,1> & xIn, dtype (&valND)[2]){
    kerValnD((dtype) (xIn-_center).norm(), valND);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<long dim>
inline void kernel<dim>::kerValnDS(const dtype xIn, dtype (&valND)[2]) {
    valND[0] = kerVal(xIn);
    valND[1] = kerValDS(xIn);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<long dim>
inline void kernel<dim>::kerValnDS(const Matrix<dtype, dim,1> & xIn, dtype (&valND)[2]){
    kerValnDS((dtype) (xIn-_center).norm(), valND);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <long dim>
inline void kernel<dim>::kerValnDeriv(const Matrix<dtype, dim, 1> &xIn, dtype (&valND)[2]) {
    _dxTemp = xIn-_center;
    kerValnDS((dtype) _dxTemp.norm(), valND);

    //Replace second value with derivative of error in coef instead of derivative of polynomial
    valND[1] = ((dtype)1.)+valND[1]*((dtype) (_dxTemp.transpose()*_trans));
}

#endif //CPP_KERNEL_H

