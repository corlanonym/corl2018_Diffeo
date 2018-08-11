//
// Created by elfuius on 08/04/18.
//

#ifndef CPP_POLYEVAL_H
#define CPP_POLYEVAL_H

#include "constants.h"

/*
 * n-order poly
 * y = coeffs[0] + coeffs[1]*x + coeffs[2]*x^2 + .. coeffs[n]*x^n
 */

inline dtype polyEval( const dtype x, const dtype * coeffs, const int n ){
    //Constant
    dtype result = coeffs[0];
    dtype xPoly = x;
    //Up to last coeff
    for (int i=1; i<n; ++i){
        result += coeffs[i]*xPoly;
        xPoly*=x;
    }
    //last coeff
    result += coeffs[n]*xPoly;
    return result;
}

inline dtype polyEvalDeriv( const dtype x, const dtype * coeffs, const int n ){
    //Constant
    dtype result = coeffs[1];
    dtype xPoly = x;
    //Up to last coeff
    for (int i=2; i<n; ++i){
        result += ((double) i)*coeffs[i]*xPoly;
        xPoly*=x;
    }
    //last coeff
    result += ((double) n)*coeffs[n]*xPoly;
    return result;
}



#endif //CPP_POLYEVAL_H
