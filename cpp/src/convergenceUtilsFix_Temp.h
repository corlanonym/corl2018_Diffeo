//
// Created by elfuius on 12/02/18.
//

#ifndef CPP_CONVERGENCEUTILSFIX_H
#define CPP_CONVERGENCEUTILSFIX_H

#include "constants.h"
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

template <long dimTot, bool doAddConv>
void ensureRadMinConv2PtrFix(const dtype * const xPtr, dtype * const vPtr){

    const dtype alpha0 = __alpha0_py;
    const Array<dtype,dimTot,1> breakRadii = (Eigen::Array<dtype, -1, -1>(dimTot,1) << __breakRadii_py).finished();

    //Get the map, this way data in vPtr will change too
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
#endif //CPP_CONVERGENCEUTILSFIX_H
