//
// Created by elfuius on 08/02/18.
//

#ifndef CPP_KERNEL_H
#define CPP_KERNEL_H

#include "constants.h"
#include <Eigen/Dense>
#include <string>
#include <iostream>

using namespace Eigen;
using namespace std;

/*
template<long dim>
const long getNParsKernel(){
    return dim+dim+3;
}
 */



template <long dim>
class kernel{
public:
    kernel(Matrix<dtype, dim,1> center, Matrix<dtype, dim, 1> trans, dtype b, dtype d, dtype e);
    kernel(const string defString);

    static long nPars(){return dim+dim+3;};

    //Number of parameters: dim+dim+3
    inline const long nPars(){return dim+dim+3;};

    inline Matrix<dtype, dim,1> getCenter()const{return _center;};
    inline const Matrix<dtype, dim,1> & getCenterR()const{return _center;};

    inline Matrix<dtype, dim,1> getTrans()const{return _trans;};
    inline const Matrix<dtype, dim,1> & getTransR()const{return _trans;};

    dtype outOfBaseCoef()const{return _y03;};
    dtype minCoef()const{return _y03;};
    dtype maxCoef()const{return _y00;};


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
    dtype _b,_d,_e;
    dtype _ddy,_y00,_dy00, _y01, _dy01, _y02, _dy02, _y03, _p1, _p2;

    dtype _tempVal;

    Matrix<dtype, dim,1> _dxTemp;

};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<long dim>
kernel<dim>::kernel(Matrix<dtype, dim, 1> center, Matrix<dtype, dim, 1> trans, dtype b, dtype d, dtype e) {
    _center = center;
    _trans = trans;
    _b = b;
    _d = d;
    _e = e;

    computeInternal();
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<long dim>
kernel<dim>::kernel(const string defString) {
    //Construct the istringstream
    istringstream defStringStream(defString);
    string::size_type sz;
    string line;

    dtype tmpVal;

    //Start with center
    for (size_t i=0; i<dim; ++i){
        getline(defStringStream, line);
        //line >> tmpVal;
        tmpVal = (dtype) stod(line, &sz);
        _center(i) = tmpVal;
    }
    //Continue with translation
    for (size_t i=0; i<dim; ++i){
        getline(defStringStream, line);
        //line >> tmpVal;
        tmpVal = (dtype) stod(line, &sz);
        _trans(i) = tmpVal;
    }
    //Rest
    getline(defStringStream, line);
    //line >> _b;
    _b = (dtype) stod(line, &sz);
    getline(defStringStream, line);
    //line >> _d;
    _d = (dtype) stod(line, &sz);
    getline(defStringStream, line);
    //line >> _e;
    _e = (dtype) stod(line, &sz);
    computeInternal();
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<long dim>
void kernel<dim>::computeInternal(){

    dtype b,d,e;
    b =_b;
    d =_d;
    e =_e;

    _p1 = (dtype) (.5-_d)*_b;
    _p2 = (dtype) (.5+_d)*_b;

    _ddy = (dtype) 4.0*(e - 1.0)/(std::pow(b, 2)*(4.0*std::pow(d, 2) - 1.0));
    _y00 = (dtype) 1.00000000000000;
    _dy00 = (dtype) 0.0;
    _y01 = (dtype) 0.5*(4.0*d - (2.0*d - 1.0)*(e - 1.0) + 2.0)/(2.0*d + 1.0);
    _dy01 = (dtype) 2.0*(e - 1.0)/(b*(2.0*d + 1.0));
    _y02 = (dtype) 0.5*(6.0*d*e - 2.0*d + e + 1.0)/(2.0*d + 1.0);
    _dy02 = (dtype) 2.0*(e - 1.0)/(b*(2.0*d + 1.0));
    _y03 = (dtype) _e;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<long dim>
inline dtype kernel<dim>::kerVal(const dtype xIn){
    //Check which case
    if(xIn>_b){
        // Constant influence outside base
        return _y03;
    }else if(xIn>_p2){
        // outer acceleration zone
        _tempVal = xIn-_p2;
        return _y02+_dy02*_tempVal+(1./2.)*(_ddy*_tempVal*_tempVal);
    }else if(xIn>_p1){
        // constant velocity zone
        return _y01+_dy01*(xIn - _p1);
    }else{
        // inner acceleration zone
        return _y00+_dy00*xIn-(1./2.)*_ddy*xIn*xIn;
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
        return (dtype) 0.;
    }else if(xIn>_p2){
        // outer acceleration zone
        _tempVal = xIn-_p2;
        return _dy02+_ddy*_tempVal;
    }else if(xIn>_p1){
        // constant velocity zone
        return _dy01;
    }else{
        // inner acceleration zone
        return _dy00-_ddy*xIn;
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

