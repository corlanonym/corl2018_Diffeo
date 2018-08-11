//
// Created by elfuius on 12/02/18.
//

#include "fileVector.h"

namespace schlepil2 {
    void WriteVector(
            const string &filename,
            const Matrix<dtype, 1, -1> &vect,
            const int prec) {
        ofstream file(filename);
        if (!file.is_open()) {
            throw runtime_error(
                    "WriteVector unable to open file: "
                    + filename);
        }
        file << vect.size();
        for (size_t i = 0; i < (size_t) vect.size(); i++) {
            file << " " << std::setprecision(prec) << vect(i);
        }
        file << std::endl;
        file.close();
    }

    void WriteMatrix(
            const string &filename,
            const Matrix<dtype, -1, -1> &mat,
            const int prec) {
        ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error(
                    "WriteMatrix unable to open file: "
                    + filename);
        }
        file << mat.rows();
        file << " " << mat.cols();
        for (size_t j = 0; j < (size_t) mat.cols(); j++) {
            for (size_t i = 0; i < (size_t) mat.rows(); i++) {
                //Store the matrix in column major order
                file << " " << std::setprecision(prec) << mat(i, j);
            }
        }
        file << std::endl;
        file.close();
    }

    void WriteMatrixPy(
            const string &filename,
            const Matrix<dtype, -1, -1> &mat,
            const int prec) {
        ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error(
                    "WriteMatrix unable to open file: "
                    + filename);
        }
        for (size_t i = 0; i < (size_t) mat.rows()-1; i++) {
            for (size_t j = 0; j < (size_t) mat.cols() - 1; j++) {
                //Write up to last value
                file << std::setprecision(prec) << mat(i, j) << " ";
            }
            //Last value and line break
            file << std::setprecision(prec) << mat(i, mat.cols() - 1) << std::endl;
        }
        //Last row
        for (size_t j = 0; j < (size_t) mat.cols() - 1; j++) {
            //Write up to last value
            file << std::setprecision(prec) << mat(mat.rows()-1, j) << " ";
        }
        //Last value
        file << std::setprecision(prec) << mat(mat.rows()-1, mat.cols() - 1);
        file.close();
    }


    Matrix<dtype, 1, -1> ReadVector(
            const string &filename) {
        dtype value;
        ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error(
                    "ReadVector unable to open file: "
                    + filename);
        }
        size_t size;
        file >> size;
        Matrix<dtype, 1, -1> vect(size);
        for (size_t i = 0; i < size; i++) {
            file >> value;
            vect(i) = value;
        }
        file.close();
        return vect;
    }

    Matrix<dtype, -1, -1> ReadMatrix(
            const std::string &filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error(
                    "ReadMatrix unable to open file: "
                    + filename);
        }
        size_t rows, cols;
        dtype value;
        file >> rows;
        file >> cols;

        Eigen::MatrixXd mat(rows, cols);
        //column major order
        for (size_t j = 0; j < cols; j++) {
            for (size_t i = 0; i < rows; i++) {
                //This is probably not very efficient...
                file >> value;
                mat(i, j) = value;
            }
        }
        file.close();
        return mat;
    }
}
