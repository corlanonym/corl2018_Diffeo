//
// Created by elfuius on 12/02/18.
//

#ifndef CPP_FILEVECTOR_H
#define CPP_FILEVECTOR_H

#include "constants.h"

#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace Eigen;
using namespace std;

namespace schlepil2 {
/**
 * Write the given vector into
 * given filename
 */
    void WriteVector(
            const string &filename,
            const Matrix<dtype, 1, -1> &vect,
            const int prec = 15);

/**
 * Write the given matrix into
 * given filename
 */
    void WriteMatrix(
            const string &filename,
            const Matrix<dtype, -1, -1> &mat,
            const int prec = 15);
/**
 * Write the given matrix into
 * given filename using standard numpy layout
 */
    void WriteMatrixPy(
            const string &filename,
            const Matrix<dtype, -1, -1> &mat,
            const int prec = 15);

/**
 * Read from given filename a Vector
 * and return it
 */
    Matrix<dtype, 1, -1> ReadVector(
            const string &filename);

/**
 * Read from given filename a Matrix
 * and return it
 */
    Matrix<dtype, -1, -1> ReadMatrix(
            const string &filename);

}
#endif //CPP_FILEVECTOR_H
