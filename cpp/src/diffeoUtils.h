//
// Created by elfuius on 08/02/18.
//

#ifndef CPP_DIFFEOUTILS_H
#define CPP_DIFFEOUTILS_H

#include "constants.h"
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <iomanip>
#include <fstream>

//Choose kernel here
//#include "c1Kernel.h"
#include "minContJerkKernel.h"

using namespace Eigen;
using namespace std;

template <long dim>
class multiTrans{
public:
    multiTrans();
    multiTrans(const string & fileName);

    //This nPars returns the number of pars per kernel
    static long nParsPerKernel(){return kernel<dim>::nPars();};
    static long nPars(){return 2;};

    void setMaxTrans(const long N);

    void addKernel( const kernel<dim> & aKernel );

    void forwardTransform(Matrix<dtype, dim, 1> & xIn);
    void forwardTransformJac(Matrix<dtype, dim, 1> & xIn, Matrix<dtype, dim, dim> & Jac, const bool outInvJac=false, const char whichSide = 'l');
    void forwardTransformV(Matrix<dtype, dim, 1> & xIn, Matrix<dtype, dim, 1> & vIn);

    void inverseTransform(Matrix<dtype, dim, 1> & xInPrime, const dtype epsInv=dtypeEpsInv);
    void inverseTransformJac(Matrix<dtype, dim, 1> & xInPrime, Matrix<dtype, dim, dim> & Jac, const dtype epsInv=dtypeEpsInv, const bool outInvJac=false, const char whichSide = 'l');
    void inverseTransformV(Matrix<dtype, dim, 1> & xInPrime, Matrix<dtype, dim, 1> & vInPrime, const dtype epsInv=dtypeEpsInv);

    // Not necessary in my code, just to be sure
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

//private:
public://Debug

    std::vector<kernel<dim>, Eigen::aligned_allocator<kernel<dim>>>_kernelList;
    Matrix<dtype, dim, 1> _dx;
    Matrix<dtype, dim, 1> _dxC;
    Matrix<dtype, dim, dim> _tmpJac;
    Matrix<dtype, dim, dim> _tmpJacV;
    Matrix<dtype, dim, 1> _xtmpForI;
    Matrix<dtype, dim, dim> _tmpJacForI;
    dtype _kerValnD[2];

    Array<dtype, 1, -1> _gammaCurrent;
    Array<dtype, 1, -1> _gammaMin;
    Array<dtype, 1, -1> _gammaMax;

    //For speed-up
    Matrix<dtype, dim, 1> _xInPrimeLast;
    Array<dtype, 1, -1> _gammaLast;

    ColPivHouseholderQR<Matrix<dtype, dim, dim>> _QRdecomp;


};

template <long dim>
class diffeomorphism{
public:
    diffeomorphism(){_epsInv=dtypeEpsInv;  _transformationList.clear();};
    diffeomorphism(const string & diffeoFile);

    //nPars forwards the needed pars per kernel
    //-> same as nPars for multitrans
    static long nParsPerKernel(){kernel<dim>::nPars();};
    static long nPars(){return 2;};

    void addTransformation( const bool direction, const multiTrans<dim> aMultiTrans );

    void forwardTransform(Matrix<dtype, dim, 1> & xIn, const long kStart=0, const long kStop=-1);
    void forwardTransformV(Matrix<dtype, dim, 1> & xIn, Matrix<dtype, dim, 1> & vIn, long kStart=0, long kStop=-1);
    void forwardTransformJac(Matrix<dtype, dim, 1> & xIn, Matrix<dtype, dim, dim> & Jac, const long kStart=0, const long kStop=-1, const bool outInvJac=false,
                             const char whichSide='l');


    void inverseTransform(Matrix<dtype, dim, 1> & xInPrime, const long kStart=0, const long kStop=-1);
    void inverseTransformV(Matrix<dtype, dim, 1> & xInPrime, Matrix<dtype, dim, 1> & vInPrime, const long kStart=0, const long kStop=-1);
    void inverseTransformJac(Matrix<dtype, dim, 1> & xInPrime, Matrix<dtype, dim, dim> & JacPrime, const long kStart=0, const long kStop=-1, const bool outInvJac=false,
                             const char whichSide='l');

//private:
public://Debug

    dtype _epsInv;
    vector<pair<bool, multiTrans<dim>>, Eigen::aligned_allocator<pair<bool, multiTrans<dim>>>>_transformationList;
    bool _mixedDirections;
    bool _allDirTrue;
    Matrix<dtype, dim,dim> _tmpJacJac;
    Matrix<dtype, dim,dim> _tmpJacV;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// IMPLEMENTATIONS MULTITRANS
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <long dim>
multiTrans<dim>::multiTrans() {
    _kernelList.clear();
    _gammaCurrent = Array<dtype,1,-1>::Zero(0);
    _gammaMin = Array<dtype,1,-1>::Zero(0);
    _gammaMax = Array<dtype,1,-1>::Zero(0);

    _gammaLast = Array<dtype,1,-1>::Zero(0);

    _xInPrimeLast.setOnes();
    _xInPrimeLast*=1.e300;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <long dim>
multiTrans<dim>::multiTrans(const string & fileName):multiTrans(){

    stringstream defStringStream;
    string partialDefString, cline;
    long dimRead;
    long nKernelTot;

    //Read
    try{
        ifstream defFile(fileName);
        //assert(defFile.is_open() && "Could not open file");
        if (!defFile.is_open()){
            throw runtime_error("file not found -> Use as defstring");
        }
        defStringStream << defFile.rdbuf();
        defFile.close();
    }catch(...){
        defStringStream = stringstream(fileName);
    }

    //multiTrans(); //Clean up
    //this->~multiTrans<dim>();
    //new (this) multiTrans();

    //Expected layout

    // dim
    // nKernelTot
    // nKernelTot * [pars per Kernel]
    defStringStream >> dimRead;
    assert( (dimRead==dim) && "incompatible dimension definition");
    defStringStream >> nKernelTot;
    assert( (nKernelTot>=0) && "Negative kernel number ?!?" );

    //Offstream leaves empty line
    getline(defStringStream, cline);//Dummy read

    //Loop
    for (long i=0; i<nKernelTot; ++i){
        partialDefString = "";
        for (long j=0; j<nParsPerKernel(); ++j){
            getline(defStringStream, cline);
            partialDefString += cline;
            partialDefString += "\n";
        }
        addKernel(kernel<dim>(partialDefString));
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <long dim>
void multiTrans<dim>::setMaxTrans(const long N) {
    _kernelList.reserve(N);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <long dim>
void multiTrans<dim>::addKernel(const kernel<dim> & aKernel){
    _kernelList.push_back(aKernel);
    _gammaCurrent.conservativeResize(_gammaCurrent.size()+1);
    _gammaLast.conservativeResize(_gammaCurrent.size()+1);
    _gammaMin.conservativeResize(_gammaMin.size()+1);
    _gammaMax.conservativeResize(_gammaMax.size()+1);

    _gammaCurrent(_kernelList.size()-1) = _kernelList.back().outOfBaseCoef();
    _gammaLast(_kernelList.size()-1) = _kernelList.back().outOfBaseCoef();
    _gammaMin(_kernelList.size()-1) = _kernelList.back().minCoef();
    _gammaMax(_kernelList.size()-1) = _kernelList.back().maxCoef();
    //std::cout << std::endl << _kernelList.back()._y03 << std::endl << _gammaMin << std::endl << _kernelList.back()._y00 << std::endl  << _gammaMax << std::endl; // dbg
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <long dim>
void multiTrans<dim>::forwardTransform(Matrix<dtype, dim, 1> &xIn) {

    _dx.setZero();
    // Add up dx
    for (typename vector<kernel<dim>, Eigen::aligned_allocator<kernel<dim>>>::iterator it = _kernelList.begin(); it != _kernelList.end(); ++it){
        _dx += it->getTransR()*it->kerVal(xIn);
    }
    // Add
    xIn += _dx;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <long dim>
void multiTrans<dim>::forwardTransformJac(Matrix<dtype, dim, 1> &xIn, Matrix<dtype, dim, dim> &Jac, const bool outInvJac, const char whichSide) {

    assert( (whichSide=='l') or (whichSide=='r'));
    if (not ((outInvJac and (whichSide=='r')) or  ((not outInvJac) and (whichSide=='l')))){
        cerr << "Except special cases whichSide should be left if outInvJac False or right and true ; " << __FILE__ << " ; " << __LINE__ << endl;
    }

    _dx.setZero();
    _tmpJac.setIdentity();

    //Add up translation and jacobian
    for (typename vector<kernel<dim>, Eigen::aligned_allocator<kernel<dim>>>::iterator it = _kernelList.begin(); it != _kernelList.end(); ++it){

        _dxC = xIn-it->getCenterR(); //Distance to center
        it->kerValnDS((dtype) _dxC.norm(), _kerValnD); // Get Value and derivative
        //Translation
        _dx += it->getTransR()*_kerValnD[0];

        //Jacobian
        //_tmpJac.noalias() = _tmpJac + (it->getTransR()*(_dxC*_kerValnD[1]).transpose());
        _tmpJac += (it->getTransR()*(_dxC*_kerValnD[1]).transpose());
    }
    if(outInvJac){
        //Apply the inversed jacobian
        _tmpJac = _tmpJac.inverse().eval();
    }

    //Apply
    xIn+=_dx;
    if (whichSide=='l'){
        Jac = _tmpJac*Jac;
    }else{
        Jac = Jac*_tmpJac;
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <long dim>
void multiTrans<dim>::forwardTransformV(Matrix<dtype, dim, 1> &xIn, Matrix<dtype, dim, 1> &vIn) {
    // Compute the forward jacobian
    _tmpJacV.setIdentity();
    forwardTransformJac(xIn, _tmpJacV, false, 'l');
    //Apply
    vIn = _tmpJacV*vIn;
    //Done
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <long dim>
void multiTrans<dim>::inverseTransform(Matrix<dtype, dim, 1> &xInPrime, const dtype epsInv) {
    const dtype convScale = 0.8;

    //old -> new below
    Array<dtype, 1, -1> gammaCurrent = _gammaCurrent; //Default -> out of base influence



    Matrix<dtype,dim,1> x;
    typename vector<kernel<dim>, Eigen::aligned_allocator<kernel<dim>>>::iterator it;
    size_t i;
    dtype errCurrent;
    bool isDone = false;

    //Get best init for speed-up
    //Distance old/new point
    dtype pointDistSq = (xInPrime-_xInPrimeLast).squaredNorm();
    it = _kernelList.begin();
    for (i=0; i<_kernelList.size(); ++i) {
        if (pointDistSq<(maxRelCloseInv*(it->_b)*(it->_b))){
            //points are close -> Use last gamma value
            gammaCurrent(i) = _gammaLast(i);
        }
        it++;
    }

    while (not isDone){
        isDone = true;
        x = xInPrime;
        //Compute current inverse
        for (i=0; i<_kernelList.size(); ++i){
            x -= _kernelList[i].getTransR()*gammaCurrent(i); //todo Check for alternatives
        }
        //Update for each kernel
        it = _kernelList.begin();
        for (i=0; i<_kernelList.size(); ++i) {
            //This incluence and derivative of the error in coef
            it->kerValnDeriv(x, _kerValnD);
            //Error
            errCurrent = gammaCurrent(i)-_kerValnD[0];
            //Check if done
            isDone = (-errCurrent<epsInv) && (errCurrent<epsInv) && isDone;
            //Update estimate
            gammaCurrent(i) -= errCurrent/_kerValnD[1]*convScale;
            //Limit
            gammaCurrent(i) = max(min(gammaCurrent(i), _gammaMax(i)), _gammaMin(i));
            it++;
        }
    }

    //Save the found gammas and the point
    _xInPrimeLast = xInPrime;
    _gammaLast = gammaCurrent;

    //Actually apply it
    for (i=0; i<_kernelList.size(); ++i){
        xInPrime -= _kernelList[i].getTransR()*gammaCurrent(i); //todo Check for alternatives
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///
/// \tparam dim
/// \param xInPrime
/// \param Jac
/// \param epsInv
/// \param outInvJac : If True, the jacobian of the corresponding forward transformation is handed back; So its the inverse of the actual jacobian (the jacobian of the inverse transformation)
/// \param whichSide
template <long dim>
void multiTrans<dim>::inverseTransformJac(Matrix<dtype, dim, 1> &xInPrime, Matrix<dtype, dim, dim> &Jac, dtype epsInv, const bool outInvJac, const char whichSide) {

    assert( (whichSide=='l') or (whichSide=='r'));
    if (not ((outInvJac and (whichSide=='r')) or  ((not outInvJac) and (whichSide=='l')))){
        cerr << "Except special cases whichSide should be left if outInvJac False or right and true ; " << __FILE__ << " ; " << __LINE__ << endl;
    }

    //Step one -> compute the inversion
    inverseTransform(xInPrime, epsInv);
    // now xInPrime is no longer in prime space

    //Compute the forward transformation to gain access to Jacobian
    // If we propagate a velocity we have to left multiply the velocity with the inverse, since vprime is in
    // vprime = Jac.v
    // So get the jacobian
    _tmpJacForI.setIdentity();
    _xtmpForI = xInPrime; //dummy variable
    //xtmpForI = xInPrime; //dummy variable
    forwardTransformJac(_xtmpForI, _tmpJacForI);
    //_tmpJacForI now holds the jacobian of the associated forward jacobian, so it's the inverse of the jacobian of this transformation

    //Inverse if necessary
    if (not outInvJac){
        _tmpJacForI = _tmpJacForI.inverse().eval();
    }

    //Multiply
    if (whichSide=='l'){
        Jac = _tmpJacForI*Jac;
    }else{
        Jac = Jac*_tmpJacForI;
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <long dim>
void multiTrans<dim>::inverseTransformV(Matrix<dtype, dim, 1> &xInPrime, Matrix<dtype, dim, 1> &vInPrime, const dtype epsInv) {
    // Compute the forward jacobian
    _tmpJacV.setIdentity();
    inverseTransformJac(xInPrime, _tmpJacV, epsInv, false, 'l');
    //Apply
    vInPrime = _tmpJacV*vInPrime;
    //Done
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// IMPLEMENTATIONS DIFFEOMORPHISM
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <long dim>
diffeomorphism<dim>::diffeomorphism(const string & diffeoFile) {
    _epsInv=dtypeEpsInv;
    _transformationList.clear();

    stringstream defStringStream;

    string partialDefString;
    string cline;

    long tmpLong;
    long nMultiTrans;
    bool tmpBool;

    // expected format
    // dimension
    // number of multitrans
    // for each multitrans
        // direction
        // dim
        // number of kernels
        // nPars multitrans + number of kernels * parsPerKernel


    try {
        ifstream defFile(diffeoFile);
        //assert(defFile.is_open() && "Could not open file");
        if (!defFile.is_open()){
            cout << defFile.is_open() << endl;
            throw runtime_error("file not found -> Use as defstring");
        }
        defStringStream << defFile.rdbuf();
        defFile.close();
    }catch(...){
        defStringStream = stringstream(diffeoFile);
    }

    defStringStream >> tmpLong;
    if (not(tmpLong==dim)){
        throw runtime_error("Incompatible dim");
    }
    defStringStream >> nMultiTrans;
    if ( not (nMultiTrans>=0)){
        throw runtime_error("Negative number of multitransformations?!");
    }

    //Assure minimal capacity
    _transformationList.reserve(nMultiTrans);

    for (long i=0; i<nMultiTrans; ++i){
        //get direction
        defStringStream >> tmpBool;

        //Offstream leaves empty line
        getline(defStringStream, cline);//DummyRead

        //Reset partial string
        partialDefString = "";

        //Get dimension
        getline(defStringStream, cline);
        partialDefString += cline;
        partialDefString += "\n";
        //Get next number of kernels
        getline(defStringStream, cline);
        partialDefString += cline;
        partialDefString += "\n";

        tmpLong = stol(cline);

        for (long j=0; j<tmpLong*multiTrans<dim>::nParsPerKernel(); ++j){
            getline(defStringStream, cline);
            partialDefString += cline;
            partialDefString += "\n";
        }
        //append
        addTransformation(tmpBool, multiTrans<dim>(partialDefString));
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <long dim>
void diffeomorphism<dim>::addTransformation( const bool direction, const multiTrans<dim> aMultiTrans ){
    _transformationList.push_back(pair<bool, multiTrans<dim>>(direction,multiTrans<dim>(aMultiTrans)));
    _mixedDirections = false;
    _allDirTrue = _transformationList[0].first;
    for (size_t i=0; i<_transformationList.size()-1;++i){
        _mixedDirections = _mixedDirections || (_transformationList[i].first != _transformationList[i+1].first);
        _allDirTrue = _allDirTrue && _transformationList[i+1].first;
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <long dim>
void diffeomorphism<dim>::forwardTransform(Matrix<dtype, dim, 1> &xIn, const long kStart, const long kStop) {
    // Succesively call all multitrans in correct order
    assert((kStart>=0) && ( (kStop==-1) || ((kStop>kStart) && (kStop<(long)_transformationList.size()) )));

    typename vector<pair<bool,multiTrans<dim>>, Eigen::aligned_allocator<pair<bool,multiTrans<dim>>>>::iterator iStart = _transformationList.begin()+kStart;
    typename vector<pair<bool,multiTrans<dim>>, Eigen::aligned_allocator<pair<bool,multiTrans<dim>>>>::iterator iEnd;

    if (kStop==-1){
        iEnd = _transformationList.end();
    }else{
        iEnd = _transformationList.begin()+kStop;
    }

    for (typename vector<pair<bool,multiTrans<dim>>, Eigen::aligned_allocator<pair<bool,multiTrans<dim>>>>::iterator it = iStart; it!=iEnd; ++it){
        if (it->first){
            it->second.forwardTransform(xIn);
        } else{
            it->second.inverseTransform(xIn, _epsInv);
        }
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <long dim>
void diffeomorphism<dim>::inverseTransform(Matrix<dtype, dim, 1> &xInPrime, const long kStart, const long kStop) {
    // Succesively call all multitrans in correct order
    assert((kStart>=0) && ( (kStop==-1) || ((kStop>kStart) && (kStop<(long)_transformationList.size()) )));

    typename vector<pair<bool,multiTrans<dim>>, Eigen::aligned_allocator<pair<bool,multiTrans<dim>>>>::reverse_iterator iStart = _transformationList.rbegin()+kStart;
    typename vector<pair<bool,multiTrans<dim>>, Eigen::aligned_allocator<pair<bool,multiTrans<dim>>>>::reverse_iterator iEnd;

    if (kStop==-1){
        iEnd = _transformationList.rend();
    }else{
        iEnd = _transformationList.rbegin()+kStop;
    }

    for (typename vector<pair<bool,multiTrans<dim>>, Eigen::aligned_allocator<pair<bool,multiTrans<dim>>>>::reverse_iterator it = iStart; it!=iEnd; ++it){
        if (not it->first){
            it->second.forwardTransform(xInPrime);
        } else{
            it->second.inverseTransform(xInPrime, _epsInv);
        }
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <long dim>
void diffeomorphism<dim>::forwardTransformJac(Matrix<dtype, dim, 1> &xIn, Matrix<dtype, dim, dim> &Jac, const long kStart, const long kStop, const bool outInvJac,
                                              const char whichSide) {

    // todo take into account directions for speedup

    assert( (whichSide=='l') or (whichSide=='r') );
    assert((kStart>=0) && ( (kStop==-1) || ((kStop>kStart) && (kStop<(long)_transformationList.size()) )));
    if (not ((outInvJac and (whichSide=='r')) or  ((not outInvJac) and (whichSide=='l')))){
        cerr << "Except special cases whichSide should be left if outInvJac False or right and true ; " << __FILE__ << " ; " << __LINE__ << endl;
    }

    typename vector<pair<bool,multiTrans<dim>>, Eigen::aligned_allocator<pair<bool,multiTrans<dim>>>>::iterator iStart = _transformationList.begin()+kStart;
    typename vector<pair<bool,multiTrans<dim>>, Eigen::aligned_allocator<pair<bool,multiTrans<dim>>>>::iterator iEnd;

    if (kStop==-1){
        iEnd = _transformationList.end();
    }else{
        iEnd = _transformationList.begin()+kStop;
    }

    _tmpJacJac.setIdentity();
    // Build the jacobian of the forward transformation (left-multiplication)
    for (typename vector<pair<bool,multiTrans<dim>>, Eigen::aligned_allocator<pair<bool,multiTrans<dim>>>>::iterator it = iStart; it!=iEnd; ++it){
        if (it->first){
            it->second.forwardTransformJac(xIn, _tmpJacJac, false, 'l');
        }else{
            it->second.inverseTransformJac(xIn, _tmpJacJac, _epsInv, false, 'l');
        }
    }
    /*
    //Here we have the forward jacobian except if _mixedDirections is false and _allDirTrue is false (Only inversetransformation have been performed
    if((not _mixedDirections) && (not _allDirTrue) && (nCols==dim)){
        Jac = Jac.inverse().eval();
    }
     */
    // We have to inverse if outInvJac
    if (outInvJac){
        _tmpJacJac = _tmpJacJac.inverse().eval();
    }

    // Multiply at the correct side with input matrix
    if (whichSide=='l'){
        Jac = _tmpJacJac*Jac;
    }else{
        Jac = Jac*_tmpJacJac;
    }
    //Done
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <long dim>
void diffeomorphism<dim>::forwardTransformV(Matrix<dtype, dim, 1> &xIn, Matrix<dtype, dim, 1> &vIn, long kStart, long kStop) {
    //todo speed up
    _tmpJacV.setIdentity();
    //Compute transformation and jacobian
    forwardTransformJac(xIn, _tmpJacV, kStart, kStop, false, 'l');
    //_tmpJacV now holds the forward jacobian
    //Transform velocity
    vIn = _tmpJacV*vIn;
    //Done
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <long dim>
void diffeomorphism<dim>::inverseTransformJac(Matrix<dtype, dim, 1> &xInPrime, Matrix<dtype, dim, dim> &JacPrime, const long kStart, const long kStop, const bool outInvJac,
                                              const char whichSide) {
    // Succesively call all multitrans in correct order
    assert( (whichSide=='l') or (whichSide=='r') );
    assert((kStart>=0) && ( (kStop==-1) || ((kStop>kStart) && (kStop<(long)_transformationList.size()) )));
    if (not ((outInvJac and (whichSide=='r')) or  ((not outInvJac) and (whichSide=='l')))){
        cerr << "Except special cases whichSide should be left if outInvJac False or right and true ; " << __FILE__ << " ; " << __LINE__ << endl;
    }

    typename vector<pair<bool,multiTrans<dim>>, Eigen::aligned_allocator<pair<bool,multiTrans<dim>>>>::reverse_iterator iStart = _transformationList.rbegin()+kStart;
    typename vector<pair<bool,multiTrans<dim>>, Eigen::aligned_allocator<pair<bool,multiTrans<dim>>>>::reverse_iterator iEnd;

    if (kStop==-1){
        iEnd = _transformationList.rend();
    }else{
        iEnd = _transformationList.rbegin()+kStop;
    }
    //todo the way the jacobians are handled is not the most efficient -> check if too costly
    // Build the forward jacobian by right multiplying the inversed subjacobians
    _tmpJacJac.setIdentity();
    for (typename vector<pair<bool,multiTrans<dim>>, Eigen::aligned_allocator<pair<bool,multiTrans<dim>>>>::reverse_iterator it = iStart; it!=iEnd; ++it){
        if (not it->first){
            it->second.forwardTransformJac(xInPrime, _tmpJacJac, true, 'r'); //Normally we need to compute the forward jacobian so we need to inverse
        }else{
            it->second.inverseTransformJac(xInPrime, _tmpJacJac, _epsInv, true, 'r');//Normally we need to compute the forward jacobian so no need to inverse
        }
    }
    if (not outInvJac){
        //We have to inverse because _tmpJacJac currently stores the jacobian of hte associated forward transformation (So the inverse of this transformation)
        _tmpJacJac = _tmpJacJac.inverse().eval();
    }
    //Multiply
    if (whichSide=='l'){
        JacPrime = _tmpJacJac*JacPrime;
    }else{
        JacPrime = JacPrime*_tmpJacJac;
    }
    //Done
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <long dim>
void diffeomorphism<dim>::inverseTransformV(Matrix<dtype, dim, 1> &xInPrime, Matrix<dtype, dim, 1> &vInPrime, const long kStart, const long kStop) {
    //todo speed up

    _tmpJacV.setIdentity();
    //Compute transformation and jacobian
    inverseTransformJac(xInPrime, _tmpJacV, kStart, kStop, false, 'l'); //Returns jacobian of this transformation
    //Apply
    vInPrime = _tmpJacV*vInPrime;
}

#endif //CPP_DIFFEOUTILS_H
