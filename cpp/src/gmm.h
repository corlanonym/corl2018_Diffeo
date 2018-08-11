//
// Created by elfuius on 05/02/18.
//

#ifndef CPP_GMM_H
#define CPP_GMM_H

#include "constants.h"
#include "gaussianKernel.h"
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <iostream>
#include <iomanip>
#include <fstream>

#include <stdexcept>
#include <assert.h>

using namespace std;

template <long nVarI, long nVarD>
class gmm{
public:
    gmm();
    gmm(string fileName);

    static long nPars(){return 4;};
    static long nParsPerKernel(){return gaussianKernel<nVarI, nVarD>::nPars();};

    void setPrior(const Matrix<dtype,1,-1> & newprior){
        assert((newprior.rows()==1) && (newprior.cols()==_kernelVector.size()));
        _prior = newprior;
        _prior /= (_prior.sum()+dtype_eps);
        _lnprior = newprior;
        _lnprior = _lnprior.array().log();

        _tmpWeights = Matrix<dtype,1,-1>::Zero(1,_kernelVector.size());
        _tmpMaps = Matrix<dtype,nVarD,-1>::Zero(nVarD,_kernelVector.size());
    }
    Matrix<dtype,1,-1> getPrior()const{return _prior;};

    unsigned long nK()const{ return _kernelVector.size();};

    void setDoCond(const bool doCond){
        _doCond=doCond;
        for (typename std::vector<gaussianKernel<nVarI,nVarD>, Eigen::aligned_allocator<gaussianKernel<nVarI,nVarD>>>::iterator it=_kernelVector.begin(); it!=_kernelVector.end();++it){
            it->setDoCond(doCond);
        }
    }
    bool getDoCond(){return _doCond;};

    void addKernel(const gaussianKernel<nVarI,nVarD> & newKernel, const dtype prior=1.){
        _kernelVector.push_back(newKernel); // make sure to push_back a copy
        // Adjust prior
        Matrix<dtype,1,-1> newprior = _prior;
        newprior.conservativeResize(1,_kernelVector.size());
        newprior[_kernelVector.size()-1] = prior;
        setPrior(newprior);
    }
    void addKernel(const Matrix<dtype,nVarI+nVarD,1> & mu, const Matrix<dtype,nVarI+nVarD,nVarI+nVarD> & Sigma, const dtype prior=1.){
        addKernel(gaussianKernel<nVarI, nVarD>(mu, Sigma, _doCond), prior);
    }
    // todo remove kernel

    // Not necessary in my code, just to be sure
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

/**
 * MAP functions for gmm
 * Eigen a specific function is provided for multiple or a single point
*/
    //template <typename Derived1, typename Derived2>
    //void evalMap1(const MatrixBase<Derived1> & x, MatrixBase<Derived2> & y);
    void evalMap1(const Matrix<dtype, nVarI, 1> & x, Matrix<dtype, nVarD, 1> & y);
    Matrix<dtype, nVarD, 1>evalMap1(const Matrix<dtype, nVarI, 1> & x){
                                                                        Matrix<dtype, nVarD, 1> y;
                                                                        evalMap1(x,y);
                                                                        return y;};
    void evalMap1Ptr(const dtype * const xPtr, dtype * const yPtr){
                                                                    Map<const Matrix<dtype, nVarI, 1>> x(xPtr);
                                                                    Map<Matrix<dtype, nVarD, 1>> y(yPtr);
                                                                    const Matrix<dtype, nVarI, 1> xD = x;
                                                                    Matrix<dtype, nVarD, 1> yD = y;
                                                                    evalMap1(xD,yD);
                                                                    y=yD;};
    void evalMapN(const Ref<const matDyn> &x, Ref<matDyn> y);
    Matrix<dtype, nVarD, -1> evalMapN(const Ref<const matDyn> &x);

    void operator()(const Matrix<dtype, nVarI, 1> & x, Matrix<dtype, nVarD, 1> & y){evalMap1(x,y);};

//private:
public: //debug
    std::vector<gaussianKernel<nVarI,nVarD>, Eigen::aligned_allocator<gaussianKernel<nVarI,nVarD>>> _kernelVector; // pushing back objects for performance
    Matrix<dtype,1,-1> _prior;
    Matrix<dtype,1,-1> _lnprior;
    bool _doCond;

    Matrix<dtype,1,-1> _tmpWeights;
    Matrix<dtype,nVarD,-1> _tmpMaps;

};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <long nVarI, long nVarD>
gmm<nVarI, nVarD>::gmm(){
    _prior=Matrix<dtype,-1,-1>::Zero(1,0);
    _lnprior=Matrix<dtype,-1,-1>::Zero(1,0);
    _doCond=true;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <long nVarI, long nVarD>
gmm<nVarI, nVarD>::gmm(string fileName) {

    string cline;
    string partialDefString;
    stringstream defStringStream;
    long nVarIC, nVarDC;
    long nKernel;
    bool doCondGMM;
    string::size_type sz;
    dtype intermediate;

    Matrix<dtype,-1,1> priorRead;

    _prior=Matrix<dtype,-1,-1>::Zero(1,0);
    _lnprior=Matrix<dtype,-1,-1>::Zero(1,0);

    try{
        std::ifstream defFile(fileName);
        if (!defFile.is_open()){
            throw runtime_error("Could not find file; trying to interpret as definition string");
        }
        defStringStream << defFile.rdbuf();
        defFile.close();
    }catch(...){
        defStringStream = stringstream(fileName);
    }


    // lines
    // nVarI
    // nvarD
    // doCond
    // nKernel
    // prior (nKernel lines)
    // \\Loop over kernels
        // nVarI
        // nvarD
        // mu_k (nVarI+nVarD)-lines
        // Sigma_k (nVarI+nVarD)^2-lines
        // doCond
    defStringStream >> nVarIC;
    defStringStream >> nVarDC;
    assert( (nVarIC==nVarI) && (nVarDC==nVarD) && "Dimensions incompatible" );
    defStringStream >> doCondGMM;
    defStringStream >> nKernel;

    priorRead.resize(nKernel,1);
    for (long i=0; i<nKernel; ++i){
        defStringStream >> intermediate;
        priorRead(i,0) = intermediate;
    }

    //Offstream leaves empty line
    getline(defStringStream, cline);//Dummy read

    //Loop over kernel assemble substrings
    for (long i=0; i<nKernel; ++i){
        partialDefString = "";
        for (long j=0; j<nParsPerKernel(); ++j){
            getline(defStringStream, cline);
            partialDefString += cline;
            partialDefString += "\n";
        }
        //Finished for this gk
        addKernel(gaussianKernel<nVarI, nVarD>(partialDefString));
    }
    //Set prioir and doCond
    setDoCond(doCondGMM);
    setPrior(priorRead);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
//template <long nVarI, long nVarD>
//template <typename Derived1, typename Derived2>
//void gmm<nVarI, nVarD>::evalMap1(const MatrixBase<Derived1> &x, MatrixBase<Derived2> &y) {
    //EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived1, nVarI, 1);
    //EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived2, nVarD, 1);
template <long nVarI, long nVarD>
void gmm<nVarI, nVarD>::evalMap1(const Matrix<dtype,nVarI,1> &x, Matrix<dtype,nVarD,1> &y) {

    for (size_t k = 0; k<_kernelVector.size(); k++){
        // Get the loglikes
        //getLogProb1(const Matrix<dtype,nVarI,1> & x, const dtype lnprior=0.);
        _tmpWeights(k) = _kernelVector[k].getLogProb1(x, _lnprior[k]);
        // Get all Maps
        // void evalMap1(const Matrix<dtype,nVarI,1> & x, Ref<Matrix<dtype,nVarI,1>> y){y = _muy+_SyxSxxI*(x-_mux);};
        _kernelVector[k].evalMap1(x, _tmpMaps.template block<nVarD,1>(0,k));
    }
    // Compute weights
    _tmpWeights = _tmpWeights.array().exp();
    //Regularize
    _tmpWeights.array() += dtype_eps;
    // Normalize weights
    _tmpWeights /= _tmpWeights.sum();
    // Apply weights to map
    _tmpMaps.array().rowwise() *= _tmpWeights.array();
    // sum over rows
    y = _tmpMaps.rowwise().sum();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <long nVarI, long nVarD>
void gmm<nVarI, nVarD>::evalMapN(const Ref<const matDyn> &x, Ref<matDyn> y) {
    assert ((x.cols() == y.cols()) && (x.rows()==nVarI) && (y.rows()==nVarD));

    const long N = x.cols();
    const long nk = nK();

    Matrix<dtype,-1,-1> tmpWeights(_kernelVector.size(), N);

    // Initialize to zero; maps will be succesively added
    y.setZero();

    // Get weights
    for (size_t k = 0; k<_kernelVector.size(); k++){
        //Get weights
        //void getWeightN(const Ref<const matDyn> & x, Ref<matDyn> out, const dtype lnprior=0.);
        //_kernelVector[k].getWeightN(x,tmpWeights.row(k),_lnprior(k)); // Fix it that this works
        _kernelVector[k].getWeightN(x,tmpWeights.block(k,0,1,N),_lnprior(k));
    }
    // Normalize weights
    tmpWeights.array() /= (tmpWeights.colwise().sum().array()+dtype_eps).replicate(nk,1).array();

    // Get all maps and apply weight
    for (size_t k = 0; k<_kernelVector.size(); k++){
        //evalMapNadd(Ref<matDyn> x, Ref<matDyn> out, Ref<matDyn> weight)
        //_kernelVector[k].evalMapNadd(x, y, tmpWeights.row(k)); // Fix it that this works
        _kernelVector[k].evalMapNadd(x, y, tmpWeights.block(k,0,1,N));
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <long nVarI, long nVarD>
Matrix<dtype, nVarD, -1> gmm<nVarI, nVarD>::evalMapN(const Ref<const matDyn> &x) {
    assert(x.rows() == nVarI);
    Matrix<dtype, nVarD, -1> y(nVarD, x.cols());
    evalMapN(x,y);
    return y;
}
#endif //CPP_GMM_H
