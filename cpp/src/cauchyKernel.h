//
// Created by elfuius on 03/02/18.
//

#ifndef CPP_CAUCHYKERNEL_H
#define CPP_CAUCHYKERNEL_H

#include "constants.h"
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/LU>

#include <stdexcept>
#include <assert.h>

#include <string>
#include <iomanip>
#include <iostream>

using namespace Eigen;
using namespace std;

template <long dim>
class cauchyKernel {
public:
    cauchyKernel(){}; //Leave all uninitialzed, could be inf or nan
    cauchyKernel(const Matrix<dtype,dim,1> & mu, const Matrix<dtype,dim,dim> & Sigma, const bool doCond=true, dtype gamma=1., dtype pi=1.);
    cauchyKernel(const string & defString);
    //gaussianKernel(const gaussianKernel & other);

    ~cauchyKernel(){};

    static long nPars(){ return dim + (dim)*(dim) + 2 + 2; };

    void setMu(const Matrix<dtype,dim,1> & mu);
    void setSigma(const Matrix<dtype,dim,dim> & Sigma);
    void setDoCond(const bool doCond){_doCond = doCond;};

    Matrix<dtype,dim,1> getMu() const { return _mu;};

    Matrix<dtype,dim,dim> getSigma() const { return _S;};
    bool getDoCond() const {return _doCond;};

    void setGamma(dtype gamma){_gamma = gamma; _gammaS = gamma*gamma;};
    void setPi(dtype pi){_pi = pi;};

    // Functions specifically designed for single point calculations
/**
 * Functions to compute the likelihood ("weight") of the gaussian
 */
    dtype getWeight1(const Matrix<dtype,dim,1> & x);
    void getWeightN(const Ref<const matDyn> & x, Ref<rVecDyn> out);
    // Not necessary in my code, just to be sure
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

//private: //When actually using
public: //Debugging
    // Variance
    Matrix<dtype, dim, dim> _S;
    Matrix<dtype, dim, dim> _SI;
    Matrix<dtype, dim, dim> _SCIdsqrt2; //Lower triangular matrix
    // mean
    Matrix<dtype, dim, 1> _mu;

    bool _doCond;

    dtype _gamma, _pi, _gammaS;
};

////////////////////////////////////////////////////////////////////
template <long dim>
cauchyKernel<dim>::cauchyKernel(const Matrix<dtype,dim,1> & mu, const Matrix<dtype,dim,dim> & Sigma, const bool doCond, dtype gamma, dtype pi){
    setMu(mu);
    setSigma(Sigma);
    setDoCond(doCond);
    setGamma(gamma);
    setPi(pi);
}
////////////////////////////////////////////////////////////////////
template <long dim>
cauchyKernel<dim>::cauchyKernel(const string &defString) {

    Matrix<dtype,dim,1> mu;
    Matrix<dtype,dim,dim> Sigma;
    bool doCond;
    dtype gamma, pi;
    long tmpLong;
    dtype tmpDtype;
    const long nVarTot = dim;


    istringstream defStringStream(defString);

    //cout << "Construct cauchy from " << endl << defString << endl << ";" << endl;

    // lines
    // nVarI
    // nvarD
    // mu_k (nVarI+nVarD)-lines
    // Sigma_k (nVarI+nVarD)^2-lines
    // doCond
    // gamma
    // pi
    defStringStream >> tmpLong;
    assert(tmpLong==dim && "Incompatible definition dim");

    //read mean
    for (long i=0; i<nVarTot;++i){
        defStringStream >> tmpDtype;
        mu(i,0) = tmpDtype;
    }
    //Read Covarianz
    // Data is expected in c-style row-major
    for (long i=0; i<nVarTot; ++i){
        for (long j=0; j<nVarTot;++j){
            defStringStream >> tmpDtype;
            Sigma(i,j) = tmpDtype;
        }
    }
    // Read cond
    defStringStream >> doCond;
    //Get gamma and pi
    defStringStream >> gamma;
    defStringStream >> pi;

    //Call constructor
    // this is some crazy shit;
    this->~cauchyKernel();
    new (this) cauchyKernel(mu, Sigma, doCond, gamma, pi);
}

////////////////////////////////////////////////////////////////////
template <long dim>
void cauchyKernel<dim>::setMu(const Matrix<dtype, dim, 1> & mu) {
    _mu = mu;
}

////////////////////////////////////////////////////////////////////
template <long dim>
void cauchyKernel<dim>::setSigma(const Matrix<dtype, dim, dim> &Sigma) {
    LLT<Matrix<dtype, dim, dim>> lltOfS;

    _S = Sigma;
    try{
        lltOfS.compute(_S);
        if (not (lltOfS.info() == Eigen::Success)){
            cout << "Sigma failed" << endl << Sigma << endl;
            throw runtime_error("Cholesky failed -> not positive?");
        }
        _SCIdsqrt2 = Matrix<dtype,dim,dim>(lltOfS.matrixL()).inverse()/sq2; //Lower triangular matrix
        _SI = Sigma.inverse();
    }catch(const std::exception &exc){
        std::cout << exc.what() << std::endl << std::flush;;
        std::cout << "DIfferent error for Sigma" << std::endl << std::flush;
        std::cout << Sigma << std::endl << std::flush;
        throw runtime_error("Sigma failed -> other error");
    }

}

////////////////////////////////////////////////////////////////////
template <long dim>
void cauchyKernel<dim>::getWeightN(const Ref<const matDyn> &x, Ref<rVecDyn> out) {
    // Compute for array
    Matrix<dtype, dim, 1> xTemp;

    for (int i=0; i<x.cols(); ++i){
        xTemp = x.col(i);
        out(0,i) = getWeight1(xTemp);
    }
}

////////////////////////////////////////////////////////////////////
template<long dim>
dtype cauchyKernel<dim>::getWeight1(const Matrix<dtype, dim, 1> &x) {
    //Computes the weight as likelihood
    dtype out = (dtype) (_SCIdsqrt2.template triangularView<Lower>()*(x-_mu)).squaredNorm();

    out = (_gamma)/(_gammaS+out);

    if(_doCond){
        out/=_pi;
    }
/*
    if (isnan(out)){
        cout << "Nan " << endl << "x " << endl << x << endl << "mu " << endl << _mu << endl << "Sigma " << endl << _S << endl << "SCIo2" << endl << _SCIdsqrt2 << endl;
    }
    if (isinf(out)){
        cout << "Inf " << endl << "x " << endl << x << endl << "mu " << endl << _mu << endl << "Sigma " << endl << _S << endl << "SCIo2" << endl << _SCIdsqrt2 << endl;
    }
*/
    return out;

}




#endif //CPP_GAUSSIANKERNEL_H
