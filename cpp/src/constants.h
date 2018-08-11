//
// Created by elfuius on 03/02/18.
//

#ifndef CPP_CONSTANTS_H
#define CPP_CONSTANTS_H

//#include<Eigen/StdVector> // no need ? http://eigen.tuxfamily.org/bz/show_bug.cgi?id=829
#include <Eigen/Dense>

#include <cmath>
#include <math.h>

namespace schlepil2{};


//datatype stuff
typedef double dtype;
const dtype dtype_eps = 1e-250;
//Minimal exactness inversion
const dtype dtypeEpsInv = 1e-7;
//Relative closeness of points for inversion
const dtype maxRelCloseInv = 0.5*0.5; // If the distance between the new inversion point and the last inversion point is less than half a base use old gamma values

#define  __forwardIntDt 2.e-4;

const dtype ctrlSpaceOriginFactor_ = 5.; //On the baxter this zone has to be fairly large.
//const dtype ctrlSpaceOriginFactor_ = 5.e4; //In simulation we can increase this value
const dtype ctrlSpaceAssumeFinishedRadius_ = .1;

typedef Eigen::Matrix<dtype,-1,-1> matDyn;
typedef Eigen::Matrix<dtype,1,-1> rVecDyn;
typedef Eigen::Matrix<dtype,-1,1> cVecDyn;
//typedef log logdtype;
//typedef exp expdtype;


const dtype sq2 = (dtype) sqrt(2.);

using namespace schlepil2;

#endif //CPP_CONSTANTS_H
