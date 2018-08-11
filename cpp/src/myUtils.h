//
// Created by elfuius on 20/04/18.
//

#ifndef CPP_MYUTILS_H
#define CPP_MYUTILS_H

#include <iostream>

#include <Eigen/Dense>
#include <cmath>

using namespace Eigen;
using namespace std;

template<typename matrixIn>
Matrix<typename matrixIn::Scalar, -1, -1> nullspace(matrixIn A, double rtol=1e-8){

    typedef typename matrixIn::Scalar thisScalar;

    int myRank=0;

    //Perform a svd
    JacobiSVD<matrixIn> mySVD(A, ComputeFullV);

    Matrix<thisScalar, -1,1> singVals = mySVD.singularValues();

    for (int i=0; i<singVals.cols(); ++i){
        if (singVals(i,0)>rtol){
            myRank+=1;
        }
    }

    cout << mySVD.matrixV().bottomRows(mySVD.matrixV().rows()-myRank).transpose() << endl;

    return Matrix<typename matrixIn::Scalar, -1, -1> ( mySVD.matrixV().bottomRows(mySVD.matrixV().rows()-myRank).transpose() );
};


#endif //CPP_MYUTILS_H
