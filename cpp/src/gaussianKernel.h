//
// Created by elfuius on 03/02/18.
//

#ifndef CPP_GAUSSIANKERNEL_H
#define CPP_GAUSSIANKERNEL_H

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

template <long nVarI, long nVarD>
class gaussianKernel {
public:
    gaussianKernel(){}; //Leave all uninitialzed, could be inf or nan
    gaussianKernel(const Matrix<dtype,nVarI+nVarD,1> & mu, const Matrix<dtype,nVarI+nVarD,nVarI+nVarD> & Sigma, const bool doCond=true);
    gaussianKernel(const string & defString);
    //gaussianKernel(const gaussianKernel & other);

    static long nPars(){ return nVarI+nVarD + (nVarI+nVarD)*(nVarI+nVarD) + 3; };

    void setMu(const Matrix<dtype,nVarI+nVarD,1> & mu);
    void setSigma(const Matrix<dtype,nVarI+nVarD,nVarI+nVarD> & Sigma);
    void setDoCond(const bool doCond){_doCond = doCond;};

    Matrix<dtype,nVarI+nVarD,1> getMu() const { return _mu;};
    Matrix<dtype,nVarI,1> getMux() const { return _mux;};
    Matrix<dtype,nVarD,1> getMuy() const { return _muy;};

    Matrix<dtype,nVarI+nVarD,nVarI+nVarD> getSigma() const { return _S;};
    Matrix<dtype,nVarI,nVarI> getSxx() const { return _Sxx;};
    bool getDoCond() const {return _doCond;};

    // Functions specifically designed for single point calculations
/**
 * Get log like of a point in X = [x',y']' space
 * @param x the point considered
 * @param lnprioir natural logarithm of the prior prob (If the Kernel is used within a GMM)
 * @return loglike of point as double
 */
    dtype getLogProb1(const Matrix<dtype,nVarI+nVarD,1> & x, const dtype lnprior=0.);
/**
 * Get log like of a point x space
 * @see getLogProb
 * @param x the point considered
 * @param lnprioir natural logarithm of the prior prob (If the Kernel is used within a GMM)
 * @return loglike of point as double
 */
    dtype getLogProb1(const Matrix<dtype,nVarI,1> & x, const dtype lnprior=0.);

    // Generic for points stored rowwise
/**
* Get log like of points in x or X space
* @see getLogProb
* @param x the point considered
* @param lnprioir natural logarithm of the prior prob (If the Kernel is used within a GMM)
* @param out Matrix<dtype,1,-1> where the results are to be stored
* @return None
* @Warning The function getLogProbN, getWeightN have to be called with dynamic blocks
*/
    void getLogProbN(const Ref<const matDyn> & x, Ref<matDyn> out, const dtype lnprior=0.);

/**
 * Functions to compute the likelihood ("weight") of the gaussian
 */
    dtype getWeight1(const Matrix<dtype,nVarI+nVarD,1> & x, const dtype lnprior=0.){ return exp(getLogProb1(x, lnprior)); };
    dtype getWeight1(const Matrix<dtype,nVarI,1> & x, const dtype lnprior=0.){ return exp(getLogProb1(x, lnprior)); };
    void getWeightN(const Ref<const matDyn> & x, Ref<matDyn> out, const dtype lnprior=0.);

/**
 * MAP maximum a posteriori: Returns the most likely y-values for a given observation in x-space
 */
    Matrix<dtype,nVarD,1> evalMap1(const Matrix<dtype,nVarI,1> & x){return _muy+_SyxSxxI*(x-_mux);};
    void evalMap1(const Matrix<dtype,nVarI,1> & x, Ref<Matrix<dtype,nVarD,1>> y){y = _muy+_SyxSxxI*(x-_mux);};
/**
 * @Warning if cpy is set to false, itself will be modified!
 */
    void evalMapN(Ref<matDyn> x, Ref<matDyn> out, bool cpy=true);
    void evalMapN(const Ref<const matDyn> & x, Ref<matDyn> out);
    void evalMapNadd(Ref<matDyn> x, Ref<matDyn> out, Ref<matDyn> weight, bool cpy);
    void evalMapNadd(const Ref<const matDyn> & x, Ref<matDyn> out, Ref<matDyn> weight);

    // Not necessary in my code, just to be sure
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

//private: //When actually using
public: //Debugging

    // Variance
    Matrix<dtype, nVarI+nVarD, nVarI+nVarD> _S;
    Matrix<dtype, nVarI, nVarI> _Sxx;
    Matrix<dtype, nVarD, nVarD> _Syy;
    Matrix<dtype, nVarD, nVarI> _Syx;
    Matrix<dtype, nVarI+nVarD, nVarI+nVarD> _SCIdsqrt2; //Lower triangular matrix
    Matrix<dtype, nVarI, nVarI> _SxxI;
    Matrix<dtype, nVarD, nVarI> _SyxSxxI;
    Matrix<dtype, nVarI, nVarI> _SxxCI; //Lower triangular matrix
    Matrix<dtype, nVarI, nVarI> _SxxCIdsqrt2; //Lower triangular matrix
    // mean
    Matrix<dtype, nVarI+nVarD, 1> _mu;
    Matrix<dtype, nVarI, 1> _mux;
    Matrix<dtype, nVarD, 1> _muy;

    dtype _scaling;
    dtype _lnscaling;

    dtype _scalingx;
    dtype _lnscalingx;

    bool _doCond;

    // For faster computation specific matrix types
    // todo

};

////////////////////////////////////////////////////////////////////
template <long nVarI, long nVarD>
gaussianKernel<nVarI, nVarD>::gaussianKernel(const Matrix<dtype,nVarI+nVarD,1> & mu, const Matrix<dtype,nVarI+nVarD,nVarI+nVarD> & Sigma, const bool doCond){
    setMu(mu);
    setSigma(Sigma);
    setDoCond(doCond);
}
////////////////////////////////////////////////////////////////////
template <long nVarI, long nVarD>
gaussianKernel<nVarI, nVarD>::gaussianKernel(const string &defString) {

    Matrix<dtype,nVarI+nVarD,1> mu;
    Matrix<dtype,nVarI+nVarD,nVarI+nVarD> Sigma;
    bool doCond;
    long tmpLong;
    dtype tmpDtype;
    const long nVarTot = nVarI+nVarD;

    istringstream defStringStream(defString);

    // lines
    // nVarI
    // nvarD
    // mu_k (nVarI+nVarD)-lines
    // Sigma_k (nVarI+nVarD)^2-lines
    // doCond
    defStringStream >> tmpLong;
    assert(tmpLong==nVarI && "Incompatible definition nVarI");
    defStringStream >> tmpLong;
    assert(tmpLong==nVarD && "Incompatible definition nVarD");

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

    //Call constructor
    // this is some crazy shit;
    this->~gaussianKernel<nVarI, nVarD>();
    new (this) gaussianKernel(mu, Sigma, doCond);
    // todo This actually works however it is not recommended; But I do not know a clean way
}

////////////////////////////////////////////////////////////////////
template <long nVarI, long nVarD>
void gaussianKernel<nVarI, nVarD>::setMu(const Matrix<dtype, nVarI + nVarD, 1> & mu) {
    _mu = mu;
    _mux = _mu.template block<nVarI,1>(0,0);
    _muy = _mu.template block<nVarD,1>(nVarI,0);
}

////////////////////////////////////////////////////////////////////
template <long nVarI, long nVarD>
void gaussianKernel<nVarI, nVarD>::setSigma(const Matrix<dtype, nVarI + nVarD, nVarI + nVarD> &Sigma) {
    LLT<Matrix<dtype, nVarD+nVarI, nVarD+nVarI>> lltOfS;

    _S = Sigma;
    try{
        lltOfS.compute(_S);
        assert( lltOfS.info() == Eigen::Success && "Cholesky failed -> not positive?" );
        _SCIdsqrt2 = Matrix<dtype,nVarI+nVarD,nVarI+nVarD>(lltOfS.matrixL()).inverse()/sq2; //Lower triangular matrix
    }catch(const std::exception &exc){
        std::cout << exc.what() << std::endl << std::flush;;
        std::cout << "DIfferent error for Sigma" << std::endl << std::flush;
        std::cout << Sigma << std::endl << std::flush;
        assert(false);
    }

    // Extract submatrices for efficient accessing
    _Sxx = _S.template block<nVarI,nVarI>(0,0);
    _Syy = _S.template block<nVarD,nVarD>(nVarI,nVarI);
    _Syx = _S.template block<nVarD,nVarI>(nVarI,0);

    _SxxI = _Sxx.inverse();
    _SxxCI = Matrix<dtype,nVarI,nVarI>(_Sxx.llt().matrixL()).inverse(); //Lower triangular matrix
    _SxxCIdsqrt2 = _SxxCI/sq2; //Lower triangular matrix

    _SyxSxxI = _Syx*_SxxI;;

    // Compute scaling (Necessary to normalize gaussian)
    _scaling = 1./sqrt( pow(2.*M_PI, nVarI+nVarD) * (dtype)_S.determinant());
    _lnscaling = log(_scaling);
    _scalingx = 1./sqrt( pow(2.*M_PI, nVarI)*(dtype)_Sxx.determinant());
    _lnscalingx = log(_scalingx);
}

////////////////////////////////////////////////////////////////////
template <long nVarI, long nVarD>
dtype gaussianKernel<nVarI, nVarD>::getLogProb1(const Matrix<dtype, nVarI + nVarD, 1> &x, const dtype lnprior) {
    if (_doCond){
        return - (dtype) (_SCIdsqrt2.template triangularView<Lower>()*(x-_mu)).squaredNorm() + (_lnscaling+lnprior);
    }else{
        return - (dtype) (_SCIdsqrt2.template triangularView<Lower>()*(x-_mu)).squaredNorm();
    }
}

////////////////////////////////////////////////////////////////////
template <long nVarI, long nVarD>
dtype gaussianKernel<nVarI, nVarD>::getLogProb1(const Matrix<dtype, nVarI, 1> &x, const dtype lnprior) {
    if (_doCond){
        return - (dtype) (_SxxCIdsqrt2.template triangularView<Lower>()*(x-_mux)).squaredNorm() + (_lnscalingx+lnprior);
    }else{
        return - (dtype) (_SxxCIdsqrt2.template triangularView<Lower>()*(x-_mux)).squaredNorm();
    }
}

////////////////////////////////////////////////////////////////////
template <long nVarI, long nVarD>
void gaussianKernel<nVarI, nVarD>::getLogProbN(const Ref<const matDyn> &x, Ref<matDyn> out, const dtype lnprior) {
    assert (x.cols() == out.cols());
    assert ((x.rows() == nVarI+nVarD)||(x.rows() == nVarI));

    // todo check how to unify
    if (x.rows() == nVarI+nVarD){
        // X-space
        if (_doCond){
            out = (-(_SCIdsqrt2.template triangularView<Lower>()*(x.colwise()-_mu)).colwise().squaredNorm()).array() + (_lnscaling+lnprior); // .noalias() why can't out be noaliased
        }else {
            out = -(_SCIdsqrt2.template triangularView<Lower>() * (x.colwise() - _mu)).colwise().squaredNorm();
        }
    } else{
        // x-space
        if (_doCond){
            out = (-(_SxxCIdsqrt2.template triangularView<Lower>()*(x.colwise()-_mux)).colwise().squaredNorm()).array() + (_lnscalingx+lnprior);
        }else {
            out = -(_SxxCIdsqrt2.template triangularView<Lower>() * (x.colwise() - _mux)).colwise().squaredNorm();
        }
    }
}

////////////////////////////////////////////////////////////////////
template <long nVarI, long nVarD>
void gaussianKernel<nVarI, nVarD>::getWeightN(const Ref<const matDyn> &x, Ref<matDyn> out, const dtype lnprior) {
    // Compute log like
    getLogProbN(x,out,lnprior);
    // Inplace exp
    out = out.array().exp();
}

////////////////////////////////////////////////////////////////////
template <long nVarI, long nVarD>
void gaussianKernel<nVarI, nVarD>::evalMapN(Ref<matDyn> x, Ref<matDyn> out, bool cpy) {
    assert ((x.rows()==nVarI) && (out.rows()==nVarI) && (x.cols()==out.cols()));
    if (cpy){
        out = (_SyxSxxI*(x.colwise()-_mux)).colwise() + _muy;
    }else{
        x.colwise() -= _mux;
        out = (_SyxSxxI*x).colwise() + _muy;
    }
}
////////////////////////////////////////////////////////////////////
template <long nVarI, long nVarD>
void gaussianKernel<nVarI, nVarD>::evalMapN(const Ref<const matDyn> &x, Ref<matDyn> out) {
    assert ((x.rows()==nVarI) && (out.rows()==nVarI) && (x.cols()==out.cols()));
    out = (_SyxSxxI*(x.colwise()-_mux)).colwise() + _muy;
}

////////////////////////////////////////////////////////////////////
template <long nVarI, long nVarD>
void gaussianKernel<nVarI, nVarD>::evalMapNadd(Ref<matDyn> x, Ref<matDyn> out, Ref<matDyn> weight, bool cpy) {
    assert((x.rows()==nVarI) && (out.rows()==nVarI) && (x.cols()==out.cols()));
    assert((weight.rows()==1) && (weight.cols()==x.cols()));
    // Code is somewhat duplicated however it is more efficient
    if (cpy){
        out.array() += (((_SyxSxxI*(x.colwise()-_mux)).colwise() + _muy).array().rowwise() * weight.array().row(0));
    }else{
        x.colwise() -= _mux;
        out.array() += (((_SyxSxxI*x).colwise() + _muy).array().rowwise() * weight.array().row(0));
    }
}
////////////////////////////////////////////////////////////////////
template <long nVarI, long nVarD>
void gaussianKernel<nVarI, nVarD>::evalMapNadd(const Ref<const matDyn>& x, Ref<matDyn> out, Ref<matDyn> weight) {
    assert((x.rows()==nVarI) && (out.rows()==nVarD) && (x.cols()==out.cols()));
    assert((weight.rows()==1) && (weight.cols()==x.cols()));
    // Code is somewhat duplicated however it is more efficient
    out.array() += (((_SyxSxxI*(x.colwise()-_mux)).colwise() + _muy).array().rowwise() * weight.array().row(0));
}



#endif //CPP_GAUSSIANKERNEL_H
