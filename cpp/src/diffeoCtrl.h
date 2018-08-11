//
// Created by elfuius on 11/02/18.
//

#ifndef CPP_DIFFEOCTRL_H
#define CPP_DIFFEOCTRL_H

//#define lenFilter 17
//#define hzFilt 60.
//#define lenFilter 25
//#define hzFilt 150.
//#define lenFilter 41
//#define lenFilter 51
//#define hzFilt 200.
#define lenFilter 26
#define hzFilt 100.
#include "constants.h"
#include <Eigen/Dense>
#include <functional>
#include "diffeoUtils.h"
#include "fileVector.h"
#include "schlepilUtils.h"
#include "convergenceUtils.h"
#include <iostream>
#include <fstream>
#include <iomanip>

//Rp=0.03, Rst=0.3; Fp=4.; Fs = 60.; eqnum = firceqrip(16,Fp/(Fs/2),[Rp Rst],'stopedge'); fvtool(eqnum,'Fs',Fs,'Color','White') % Visualize filter
//Rp=0.005, Rst=0.3; Fp=6.; Fs = 100.; eqnum = firceqrip(22,Fp/(Fs/2),[Rp Rst],'stopedge'); fvtool(eqnum,'Fs',Fs,'Color','White') % Visualize filter
//Ugly but hopefully working on the robot
//#include "gmm.h"
//#include "convergenceUtils.h"



using namespace Eigen;
using namespace std;

//We consider autonous systems with or without internal variables
template <long dimTot, long dimInt>
class diffeoCtrl{
public:
    //diffeoCtrl(){_t=0.;_scaling.setOnes();_scalingInverse.setOnes(); _demSpaceOffset.setZero(); thisGetCtrlSpaceVelPtr = std::bind(&diffeoCtrl<dimTot, dimInt>::getCtrlSpaceVelPtr, this, std::placeholders::_1, std::placeholders::_2);};
    diffeoCtrl(){std::cout << "a" << std::endl;_t=0.;_scaling.setOnes();_scalingInverse.setOnes(); _demSpaceOffset.setZero();std::cout << "c" << std::endl;};
    diffeoCtrl(const string fileName);

    //Reset external time
    void reset(){_t=0.;};

    //Set scaling
    void setScaling(Matrix<dtype,dimTot,1> newScaling){_scaling=newScaling; _scalingInverse = 1./_scaling;};

    // "Forwarded" diffeo functions
    // Attention: Forward is from dem 2 ctrl
    void forwardTransform(Matrix<dtype, dimTot, 1> & xIn, const long kStart=0, const long kStop=-1);
    void forwardTransformV(Matrix<dtype, dimTot, 1> & xIn, Matrix<dtype, dimTot, 1> & vIn, long kStart=0, long kStop=-1);
    void forwardTransformJac(Matrix<dtype, dimTot, 1> & xIn, Matrix<dtype, dimTot, dimTot> & Jac, long kStart=0, long kStop=-1, const bool outInvJac=false,
                             const char whichSide='l');

    void inverseTransform(Matrix<dtype, dimTot, 1> & xInPrime, const long kStart=0, const long kStop=-1);
    void inverseTransformV(Matrix<dtype, dimTot, 1> & xInPrime, Matrix<dtype, dimTot, 1> & vInPrime, const long kStart=0, const long kStop=-1);
    void inverseTransformJac(Matrix<dtype, dimTot, 1> & xInPrime, Matrix<dtype, dimTot, dimTot> & JacPrime, const long kStart=0, const long kStop=-1, const bool outInvJac=false, const char whichSide='l');

    //Set all in demonstration
    // Forward computes the matrix from dem 2 ctrl
    // However _cStateJac is defined as being the jacobian from ctrl 2 dem -> Inverse
    void setPosDem(const Matrix<dtype, dimTot,1> & newPos){ _cStateDem = newPos; _cStateCtrl=_cStateDem;
                                                            _cStateJac.setIdentity();
                                                            forwardTransformJac(_cStateCtrl, _cStateJac, 0,-1,true,'r');};
    //Set the non-internal part in demonstration
    void setPosDem(const Matrix<dtype, dimTot-dimInt,1> & newPos){  _cStateDem.template block<dimTot-dimInt,1>(0,0) = newPos; _cStateCtrl=_cStateDem;
                                                                    _cStateJac.setIdentity();
                                                                    forwardTransformJac(_cStateCtrl,_cStateJac,0,-1,true,'r');};
    // Set all in ctrl
    // Inverse transform goes from ctrl 2 dem, so no jacobian inversion needed
    void setPosCtrl(const Matrix<dtype, dimTot,1> & newPos){_cStateCtrl = newPos; _cStateDem=_cStateCtrl;
                                                            _cStateJac.setIdentity();
                                                            inverseTransformJac(_cStateDem, _cStateJac,0,-1,false,'l');};
    // Set non-internal in ctrl
    void setPosCtrl(const Matrix<dtype, dimTot-dimInt,1> & newPos){_cStateCtrl.template block<dimTot-dimInt,1>(0,0) = newPos; _cStateDem=_cStateCtrl;
                                                                   _cStateJac.setIdentity();
                                                                   inverseTransformJac(_cStateDem, _cStateJac,0,-1,false,'l');};

    //The same as above as reference getters
    template<typename thisDType>
    void getPosDem(Matrix<thisDType, dimTot-dimInt, 1> & out)const{out = _cStateDem.template block<dimTot-dimInt, 1>(0,0).template cast<thisDType>();};
    template<typename thisDType>
    void getPosDem(Matrix<thisDType, dimTot, 1> & out)const{out = _cStateDem.template cast<thisDType>();};
    template<typename thisDType>
    void getPosCtrl(Matrix<thisDType, dimTot-dimInt, 1> & out)const{out = _cStateCtrl.template block<dimTot-dimInt, 1>(0,0).template cast<thisDType>();};
    template<typename thisDType>
    void getPosCtrl(Matrix<thisDType, dimTot, 1> & out)const{out = _cStateCtrl.template cast<thisDType>();};

    // Computes forward traj from current point
    Matrix<dtype,dimTot,-1> getTraj(const Matrix<dtype,1,-1> & tVec, bool outInDem=true);

    // Ctrl Space vel
    // First call dynClass (gmm) then assure convergence)
    //void getCtrlSpaceVel(const Matrix<dtype,dimTot,1> & xCtrl, Matrix<dtype, dimTot, 1> & vCtrl){ _dynClass(xCtrl, vCtrl);
    //                                                                                              ensureRadMinConv2(xCtrl, vCtrl, _thisConvPars);};

    //void operator()(const Matrix<dtype,dimTot,1> & xCtrl, Matrix<dtype, dimTot, 1> & vCtrl){getCtrlSpaceVel(xCtrl, vCtrl);};

    // Used in the loop
    void getNextVel(const Matrix<dtype,dimTot-dimInt,1> & posMeas, const dtype cTime, Matrix<dtype,dimTot,1>& demSpaceVel);
    void getNextVelNAcc(const Matrix<dtype,dimTot-dimInt,1> & posMeas, const dtype cTime, Matrix<dtype,dimTot,1>& demSpaceVel, Matrix<dtype,dimTot,1>& demSpaceAcc);

    //Used for python interface
    void getNextVelNAcc(const Matrix<dtype,dimTot-dimInt,1> & posMeas, const dtype cTime, Matrix<dtype,dimTot,1>& demSpaceVel, Matrix<dtype,dimTot,1>& demSpaceAcc, Matrix<dtype,dimTot,1>& ctrlSpaceVel, Matrix<dtype,dimTot,1>& ctrlSpaceAcc);


    //void getCtrlSpaceVelPtr(const dtype * const x, dtype * const v){ this->_fDyn(x,v); this->_fConv(x,v);};



    //Necessary function
    // Eval dynamic in ctrl space
    // First argument is the point cosidered in control space, second argument is where to store velocity
    // std::function is not the way to go for performance
    // However alternatives seem to be annoying ->

    // todo
    //function<void(const Matrix<dtype,dimTot,1> &, Matrix<dtype,dimTot,1> &)> _fDyn;
    //function<void(const Matrix<dtype,dimTot,1> &, Matrix<dtype,dimTot,1> &)> _fConv;
    function<void(const dtype * const, dtype * const)> _fDyn;
    function<void(const dtype * const, dtype * const)> _fConv;

    //function<void(const dtype * const, dtype * const)> thisGetCtrlSpaceVelPtr;

    //void setDyn(dynClassT aDyn){_dynClass = aDyn;};
    // Not necessary in my code, just to be sure
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

//private:
public: //DBG
    //Demonstration space offset: Vector from origin to actual target
    Matrix<dtype,dimTot,1> _demSpaceOffset;

    // Current state and "real world time"
    Matrix<dtype,dimTot,1> _cStateDem; //Demonstration space
    Matrix<dtype,dimTot,1> _cStateCtrl; //Ctrl space
    Matrix<dtype,dimTot,dimTot> _cStateJac; //Jacobian from ctrl to demonstration space for current state
    Matrix<dtype,dimTot,dimTot> _jacTmp; //Dummy variable
    Matrix<dtype,dimTot,1> _velTmpCtrl; //Ctrl space velocity as temp var
    dtype _t;
    diffeomorphism<dimTot> _diffeo;
    bool _direction; //Direction of the diffeo: If true: forward is from demonstration to control; (So false means that the inversetransformation is applied from demonstration to control)

    Array<dtype,dimTot,1> _scaling;
    Array<dtype,dimTot,1> _scalingInverse;

    Matrix<dtype,dimTot,dimTot> _tmpJacJac;
    Matrix<dtype,dimTot,dimTot> _tmpJacV;

    // Ugly but hopefully correct on robot
    //convPars<dimTot> _thisConvPars;
    //dynClassT _dynClass;

    //Hard-coded stuff for filtering
    dtype _timeStepFilt; //We use a 9 order digital fir with 15Hz cutoff and 40Hz acquisition
    Array<dtype, dimTot, lenFilter> _tempArrayVfilt;
    Array<dtype, dimTot, lenFilter> _tempArrayAccfilt;
    //const Array<dtype, 1, 9> _filtCoefs = (Array<dtype, -1, -1>(1,9) << -0.0334441146234232, 0.067167119667256, 0.149780624305019, 0.232832880332744, 0.267326980636808, 0.232832880332744, 0.149780624305019, 0.067167119667256, -0.0334441146234232).finished();
    //const Array<dtype, 1, 9> _filtCoefs = (Array<dtype, -1, -1>(1,9) << -0.0738369825933533, 0.118158422468126, -0.1232121811939, 0.178921210883778, 0.838257594278315, 0.178921210883778, -0.1232121811939, 0.118158422468126, -0.0738369825933533).finished();
    Array<dtype, 1, lenFilter> _filtCoefs;
    //const Array<dtype, 1, lenFilter> _filtCoefs = (Array<dtype, -1, -1>(1,lenFilter) << -0.0614038811017469,0.191660042569418,0.0836720034582785,0.0561258235119308,0.0505639629545392,0.0506668631280955,0.0518473765147696,0.052797270790556,0.0531410763483191,0.052797270790556,0.0518473765147696,0.0506668631280955,0.0505639629545392,0.0561258235119308,0.0836720034582785,0.191660042569418,-0.0614038811017469).finished();


    constexpr static const int _nStepHalf = (lenFilter-1)/2;

};


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// IMPLEMENTATIONS DIFFEOCTRL
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <long dimTot, long dimInt>
diffeoCtrl<dimTot, dimInt>::diffeoCtrl(const string fileName) {
    std::cout << "b" << std::endl;
    _t=0.;
    string cline;
    ifstream defFile(fileName);
    string::size_type sz;
    dtype tmpVal;
    long tmpLong;
    bool tmpBool;

    Matrix<dtype,dimTot,1> tmpScale;

    //dimTot
    //dimInt
    //scaling
    if (defFile.is_open()){
        defFile >> tmpLong;
        assert(tmpLong==dimTot);
        defFile >> tmpLong;
        assert(tmpLong==dimInt);
        defFile >> tmpBool;
        _direction = tmpBool;
        cout << "Direction set to " << _direction << endl;

        for (size_t i=0; i<dimTot; ++i){
            defFile >> tmpVal;
            tmpScale(i) = tmpVal;
        }
        setScaling(tmpScale);
        //Get the offset
        for (size_t i=0; i<dimTot; ++i){
            defFile >> tmpVal;
            _demSpaceOffset(i) = tmpVal;
        }
    }else{
        throw runtime_error("Definition file could not be opened!");
    }
    cout << "zero" << endl;
    _timeStepFilt = 1./hzFilt;
    cout << "one" << endl;
    //_filtCoefs = (Array<dtype, -1, -1>(1,lenFilter) << -0.125432630606592,0.0466906788103893,0.0435308354491001,0.043326170099747,0.0451732517301919,0.0483412509404134,0.0522290697684076,0.0563382270494426,0.060256831458649,0.0636510870430851,0.0662616207605239,0.0679025860569225,0.0684620428794382,0.0679025860569225,0.0662616207605239,0.0636510870430851,0.060256831458649,0.0563382270494426,0.0522290697684076,0.0483412509404134,0.0451732517301919,0.043326170099747,0.0435308354491001,0.0466906788103893,-0.125432630606592).finished();
    //_filtCoefs = (Array<dtype, -1, -1>(1,lenFilter) << 0.0171329934130508,-0.142868496638361,-0.00526349450176561,0.0116283812364722,0.0147313833031776,0.016752919792703,0.0191319151844839,0.0219559115190576,0.0251471401308657,0.0286051394482361,0.0322249707545332,0.0359008484999206,0.0395282427662262,0.0430059403484236,0.0462381670857469,0.0491367061229312,0.0516229324518697,0.053629684472778,0.0551029013885753,0.056002965197839,0.0563056960464728,0.056002965197839,0.0551029013885753,0.053629684472778,0.0516229324518697,0.0491367061229312,0.0462381670857469,0.0430059403484236,0.0395282427662262,0.0359008484999206,0.0322249707545332,0.0286051394482361,0.0251471401308657,0.0219559115190576,0.0191319151844839,0.016752919792703,0.0147313833031776,0.0116283812364722,-0.00526349450176561,-0.142868496638361,0.0171329934130508).finished();
    //_filtCoefs = (Array<dtype, -1, -1>(1,lenFilter) << -0.132160458931004,0.033788329927963,0.0304361828377374,0.0278691083604576,0.0259516728189559,0.0245701179729188,0.023628662353659,0.0230464468867365,0.0227550150445496,0.0226962363790755,0.0228205977222113,0.0230857991509752,0.0234556024479755,0.023898888618121,0.0243888883609498,0.0249025555019153,0.0254200584694723,0.0259243691456734,0.026400931964103,0.0268373991034463,0.027223420130592,0.0275504765698129,0.0278117536865042,0.0280020433362128,0.0281176730942941,0.0281564580933832,0.0281176730942941,0.0280020433362128,0.0278117536865042,0.0275504765698129,0.027223420130592,0.0268373991034463,0.026400931964103,0.0259243691456734,0.0254200584694723,0.0249025555019153,0.0243888883609498,0.023898888618121,0.0234556024479755,0.0230857991509752,0.0228205977222113,0.0226962363790755,0.0227550150445496,0.0230464468867365,0.023628662353659,0.0245701179729188,0.0259516728189559,0.0278691083604576,0.0304361828377374,0.033788329927963,-0.132160458931004).finished();
    _filtCoefs = (Array<dtype, -1, -1>(1,lenFilter) << 0.0786413607544648,-0.107254025879759,-0.051721960561206,-0.0190625010131211,0.00298283026016802,0.0209422731635434,0.0378796067397489,0.054764466699979,0.0713334146587406,0.0866624864655679,0.0995718189234907,0.108923200113676,0.113837029674706,0.113837029674706,0.108923200113676,0.0995718189234907,0.0866624864655679,0.0713334146587406,0.054764466699979,0.0378796067397489,0.0209422731635434,0.00298283026016802,-0.0190625010131211,-0.051721960561206,-0.107254025879759,0.0786413607544648).finished();
    cout << "two" << endl;

    // Init function
    //thisGetCtrlSpaceVelPtr = std::bind(&diffeoCtrl<dimTot, dimInt>::getCtrlSpaceVelPtr, this, std::placeholders::_1, std::placeholders::_2);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <long dimTot, long dimInt>
void diffeoCtrl<dimTot, dimInt>::forwardTransform(Matrix<dtype, dimTot, 1> &xIn, const long kStart, const long kStop) {
    /*
     * Forward transform: From dem to ctrl
     */
    //Scale
    xIn -= _demSpaceOffset;
    xIn.array() *= _scaling;
    if(_direction){
        _diffeo.forwardTransform(xIn, kStart, kStop);
    }else{
        _diffeo.inverseTransform(xIn, kStart, kStop);
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <long dimTot, long dimInt>
void diffeoCtrl<dimTot, dimInt>::forwardTransformJac(Matrix<dtype, dimTot, 1> &xIn, Matrix<dtype, dimTot, dimTot> &Jac, long kStart, long kStop, const bool outInvJac,
                                                    const char whichSide) {
    /*
     * Forward transform: From dem to ctrl
     */
    //Attention, the jacobians do not take into the scaling
    //Check
    assert((whichSide=='l') or (whichSide=='r'));
    if (not ((outInvJac and (whichSide=='r')) or  ((not outInvJac) and (whichSide=='l')))){
        cerr << "Except special cases whichSide should be left if outInvJac False or right and true ; " << __FILE__ << " ; " << __LINE__ << endl;
    }

    // Scale
    xIn -= _demSpaceOffset;
    xIn.array() *= _scaling;
    _tmpJacJac.setIdentity();

    if(_direction){
        _diffeo.forwardTransformJac(xIn, _tmpJacJac, kStart, kStop, outInvJac, whichSide); //This outputs jac from dem to ctrl
    }else{
        _diffeo.inverseTransformJac(xIn, _tmpJacJac, kStart, kStop, outInvJac, whichSide); //This automatically outputs the inverse
    }

    // Now the "correct" (with respect to outInvJac) jacobian is stored in _tmpJacJac
    // Multiply
    if (whichSide=='l'){
        Jac = _tmpJacJac*Jac;
    }else{
        Jac = Jac*_tmpJacJac;
    }
    //Done
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <long dimTot, long dimInt>
void diffeoCtrl<dimTot, dimInt>::forwardTransformV(Matrix<dtype, dimTot, 1> &xIn, Matrix<dtype, dimTot, 1> &vIn, long kStart, long kStop) {

    // Compute jacobian first
    _tmpJacV.setIdentity();
    forwardTransformJac(xIn, _tmpJacV, kStart, kStop, false, 'l');

    //Scale velocity
    vIn.array() *= _scaling;
    //Apply
    vIn = _tmpJacV*vIn;
    //Done
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <long dimTot, long dimInt>
void diffeoCtrl<dimTot, dimInt>::inverseTransform(Matrix<dtype, dimTot, 1> &xInPrime, const long kStart, const long kStop) {
    /*
     * Inverse transform: From ctrl to dem
     */
    if(not _direction){
        _diffeo.forwardTransform(xInPrime, kStart, kStop);
    }else{
        _diffeo.inverseTransform(xInPrime, kStart, kStop);
    }
    xInPrime.array() *= _scalingInverse;
    xInPrime += _demSpaceOffset;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <long dimTot, long dimInt>
void diffeoCtrl<dimTot, dimInt>::inverseTransformJac(Matrix<dtype, dimTot, 1> &xInPrime, Matrix<dtype, dimTot, dimTot> &JacPrime, const long kStart, const long kStop, const bool outInvJac,
                                                    const char whichSide) {
    /*
     * Inverse transform: From ctrl to dem
     */
    // Attention jacobians within diffeoCtrl are always from ctrl to demonstration
    // Attention, the jacobians do not take into the scaling
    // Attention jacobian has to be identity on entry
    //Velocities are modified regularly

    //Check
    assert((whichSide=='l') or (whichSide=='r'));
    if (not ((outInvJac and (whichSide=='r')) or  ((not outInvJac) and (whichSide=='l')))){
        cerr << "Except special cases whichSide should be left if outInvJac False or right and true ; " << __FILE__ << " ; " << __LINE__ << endl;
    }

    //Set
    _tmpJacJac.setIdentity();

    if(not _direction){
        _diffeo.forwardTransformJac(xInPrime, _tmpJacJac, kStart, kStop, outInvJac, whichSide);
    }else{
        _diffeo.inverseTransformJac(xInPrime, _tmpJacJac, kStart, kStop, outInvJac, whichSide); //This automatically outputs the inverse
    }

    // Now the "correct" (with respect to outInvJac) jacobian is stored in _tmpJacJac
    // Multiply
    if (whichSide=='l'){
        JacPrime = _tmpJacJac*JacPrime;
    }else{
        JacPrime = JacPrime*_tmpJacJac;
    }
    //Scale the point
    xInPrime.array() *= _scalingInverse;
    xInPrime += _demSpaceOffset;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <long dimTot, long dimInt>
void diffeoCtrl<dimTot, dimInt>::inverseTransformV(Matrix<dtype, dimTot, 1> &xInPrime, Matrix<dtype, dimTot, 1> &vInPrime, const long kStart, const long kStop) {
    //Compute jacobian first
    _tmpJacV.setIdentity();
    inverseTransformJac(xInPrime, _tmpJacV, kStart, kStop, false, 'l');
    //Apply
    vInPrime = _tmpJacV*vInPrime;
    //Scale
    vInPrime.array() *= _scalingInverse;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <long dimTot, long dimInt>
Matrix<dtype, dimTot, -1> diffeoCtrl<dimTot, dimInt>::getTraj(const Matrix<dtype, 1, -1> & tVec, bool outInDem) {

    /*
     * Compute forward trajectory for current time and position; will leave the object unchanged
     */

    Matrix<dtype, dimTot, -1> xOut = Matrix<dtype, dimTot, -1>::Zero(dimTot, tVec.cols());
    //Matrix<dtype, dimTot, 1> xC = _cStateCtrl;
    dtype xCPtr[dimTot]; //Get array
    Map<Matrix<dtype, dimTot, 1>> xC(xCPtr); //Get map object
    xC = _cStateCtrl;
    Matrix<dtype, dimTot, 1> xtmp;

    dtype tTraj = _t;

    assert(tVec[0]>=tTraj);

    for (size_t i=0; i<tVec.cols(); ++i){
        simpleIntegratePtr<dimTot>((dtype) tVec[i]-tTraj, xCPtr, _fDyn, _fConv);
        //simpleIntegrate<dimTot>((dtype) tVec[i]-tTraj, xC, _fDyn, _fConv);
        //simpleIntegrate<dimTot, diffeoCtrl<dimTot, dimInt>>((dtype) tVec[i]-tTraj, xC, *this);
        if (outInDem){
            xtmp = xC;
            inverseTransform(xtmp);
            xOut.col(i) = xtmp;
        }else{
            xOut.col(i) = xC;
        }
        tTraj = tVec[i];
    }
    return  xOut;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <long dimTot, long dimInt>
void diffeoCtrl<dimTot, dimInt>::getNextVel(const Matrix<dtype, dimTot - dimInt, 1> &posMeas, const dtype cTime, Matrix<dtype,dimTot,1>& demSpaceVel) {
    assert(cTime>=_t);

    //Take detour with pointer
    dtype xCPtr[dimTot]; //Get array
    Map<Matrix<dtype, dimTot, 1>> xC(xCPtr); //Get map object
    xC = _cStateCtrl;

    dtype vCPtr[dimTot]; //Get array
    Map<Matrix<dtype, dimTot, 1>> vC(vCPtr); //Get map object

    //Integrate till new timepoint
    simpleIntegratePtr<dimTot>((dtype) (cTime-_t), xCPtr, _fDyn, _fConv);
    //simpleIntegratePtr1<dimTot>((dtype) (cTime-_t), xCPtr, thisGetCtrlSpaceVelPtr);
    //Copy back
    _cStateCtrl = xC;
    //simpleIntegrate<dimTot>((dtype) (cTime-_t), _cStateCtrl, _fDyn, _fConv);
    //simpleIntegrate<dimTot, diffeoCtrl<dimTot, dimInt>>((dtype) (cTime-_t), _cStateCtrl, *this);

    //Keep demspace and control space consistent
    _cStateDem = _cStateCtrl;
    inverseTransform(_cStateDem); // -> ctrl 2 dem

    //Update the obtained point with new info from "reality"
    setPosDem(posMeas);

    //Get the velocity (ctrl space)
    //_fDyn(_cStateCtrl,_velTmpCtrl);
    //_fConv(_cStateCtrl,_velTmpCtrl);
    //getCtrlSpaceVelPtr(xCPtr, vCPtr);
    _fDyn(xCPtr, vCPtr);
    _fConv(xCPtr, vCPtr);
    //(*this)(_cStateCtrl,_velTmpCtrl);

    //Transform into demonstration space
    demSpaceVel = _scalingInverse*( (_cStateJac*vC).array() );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <long dimTot, long dimInt>
void diffeoCtrl<dimTot, dimInt>::getNextVelNAcc(const Matrix<dtype, dimTot - dimInt, 1> &posMeas, const dtype cTime, Matrix<dtype,dimTot,1>& demSpaceVel, Matrix<dtype,dimTot,1>& demSpaceAcc) {
    assert(cTime>=_t);

    const dtype dTAcc = (dtype) __forwardIntDt;

    //Set up all ptrs and maps
    dtype xCPtr[dimTot]; //Get array
    Map<Matrix<dtype, dimTot, 1>> xC(xCPtr); //Get map object
    xC = _cStateCtrl;
    Matrix<dtype, dimTot, 1> xCM;

    dtype vCPtr[dimTot]; //Get array
    Map<Matrix<dtype, dimTot, 1>> vC(vCPtr); //Get map object
    Matrix<dtype, dimTot, 1> vCM;

    dtype cStateTempPtr[dimTot];
    Map<Matrix<dtype, dimTot, 1>> cStateTemp(cStateTempPtr); //Get map object
    Matrix<dtype, dimTot, 1> cStateTempM; //Arg

    dtype cVelTempPtr[dimTot];
    Map<Matrix<dtype, dimTot, 1>> cVelTemp(cVelTempPtr); //Get map object
    Matrix<dtype, dimTot, 1> cVelTempM; //Arg


    //Integrate till new timepoint
    simpleIntegratePtr<dimTot>((dtype) (cTime-_t), xCPtr, _fDyn, _fConv);
    //simpleIntegratePtr1<dimTot>((dtype) (cTime-_t), xCPtr, thisGetCtrlSpaceVelPtr);
    _cStateCtrl = xC; //Copy back
    //simpleIntegrate<dimTot>((dtype) (cTime-_t), _cStateCtrl, _fDyn, _fConv);
    //simpleIntegrate<dimTot, diffeoCtrl<dimTot, dimInt>>((dtype) (cTime-_t), _cStateCtrl, *this);
    // Set new internal point
    _t = cTime;

    //Keep demspace and control space consistent
    _cStateDem = _cStateCtrl;
    inverseTransform(_cStateDem); // -> ctrl 2 dem
    //Update the obtained point with new info from "reality"
    setPosDem(posMeas);

    //Current position
    xC = _cStateCtrl; //Rest
    _fDyn(xCPtr,vCPtr);
    _fConv(xCPtr,vCPtr);
    //Transform into dem space
    xCM = xC;
    vCM = vC;
    inverseTransformV(xCM, vCM);
    // Continue to get acc
    cStateTemp = xC;
    simpleIntegratePtr<dimTot>(dTAcc, cStateTempPtr, _fDyn, _fConv);
    // Get next vel in ctrl
    _fDyn(cStateTempPtr,cVelTempPtr);
    _fConv(cStateTempPtr,cVelTempPtr);
    // to dem
    cStateTempM = cStateTemp;
    cVelTempM = cVelTemp;
    inverseTransformV(cStateTempM, cVelTempM);

    //Save
    _tempArrayVfilt.col(_nStepHalf) = vCM;
    _tempArrayAccfilt.col(_nStepHalf) = (cVelTempM-vCM)/dTAcc;

    // Do the positive time-steps
    for (int i=0; i<_nStepHalf; ++i){
        // Integrate one tFilt
        simpleIntegratePtr<dimTot>(_timeStepFilt, xCPtr, _fDyn, _fConv);
        // Get this velocity
        _fDyn(xCPtr,vCPtr);
        _fConv(xCPtr,vCPtr);
        //Transform into dem space
        xCM = xC;
        vCM = vC;
        inverseTransformV(xCM, vCM);
        // Continue to get acc
        cStateTemp = xC;
        simpleIntegratePtr<dimTot>(dTAcc, cStateTempPtr, _fDyn, _fConv);
        // Get next vel in ctrl
        _fDyn(cStateTempPtr,cVelTempPtr);
        _fConv(cStateTempPtr,cVelTempPtr);
        // to dem
        cStateTempM = cStateTemp;
        cVelTempM = cVelTemp;
        inverseTransformV(cStateTempM, cVelTempM);

        //Save
        _tempArrayVfilt.col(_nStepHalf+1+i) = vCM;
        _tempArrayAccfilt.col(_nStepHalf+1+i) = (cVelTempM-vCM)/dTAcc;
    }
    // Do the negative time-steps
    //Reset pos
    xC = _cStateCtrl; //Rest
    for (int i=0; i<_nStepHalf; ++i){
        // Integrate one tFilt
        simpleIntegratePtr<dimTot>(-_timeStepFilt, xCPtr, _fDyn, _fConv);
        // Get this velocity
        _fDyn(xCPtr,vCPtr);
        _fConv(xCPtr,vCPtr);
        //Transform into dem space
        xCM = xC;
        vCM = vC;
        inverseTransformV(xCM, vCM);
        // Continue to get acc
        cStateTemp = xC;
        simpleIntegratePtr<dimTot>(dTAcc, cStateTempPtr, _fDyn, _fConv);
        // Get next vel in ctrl
        _fDyn(cStateTempPtr,cVelTempPtr);
        _fConv(cStateTempPtr,cVelTempPtr);
        // to dem
        cStateTempM = cStateTemp;
        cVelTempM = cVelTemp;
        inverseTransformV(cStateTempM, cVelTempM);

        //Save
        _tempArrayVfilt.col(_nStepHalf-1-i) = vCM;
        _tempArrayAccfilt.col(_nStepHalf-1-i) = (cVelTempM-vCM)/dTAcc;
    }

    //Do the actual filtering
    /*
    cout << _tempArrayVfilt << endl << endl;
    cout << _tempArrayVfilt.rowwise()*_filtCoefs << endl << endl;
    cout << (_tempArrayVfilt.rowwise()*_filtCoefs).rowwise().sum() << endl << endl;
    */
    demSpaceVel = (_tempArrayVfilt.rowwise()*_filtCoefs).rowwise().sum();//_tempArrayVfilt.col(_nStepHalf);//_tempArrayVfilt.rowwise().mean();//
    demSpaceAcc = (_tempArrayAccfilt.rowwise()*_filtCoefs).rowwise().sum();//_tempArrayAccfilt.col(_nStepHalf);//_tempArrayAccfilt.rowwise().mean();//
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// void getNextVelNAcc(const Matrix<dtype,dimTot-dimInt,1> & posMeas, const dtype cTime, Matrix<dtype,dimTot,1>& demSpaceVel, Matrix<dtype,dimTot,1>& demSpaceAcc, Matrix<dtype,dimTot,1>& ctrlSpaceVel, Matrix<dtype,dimTot,1>& ctrlSpaceAcc);
template <long dimTot, long dimInt>
void diffeoCtrl<dimTot, dimInt>::getNextVelNAcc(const Matrix<dtype, dimTot - dimInt, 1> &posMeas, const dtype cTime, Matrix<dtype,dimTot,1>& demSpaceVel, Matrix<dtype,dimTot,1>& demSpaceAcc, Matrix<dtype,dimTot,1>& ctrlSpaceVel, Matrix<dtype,dimTot,1>& ctrlSpaceAcc) {
    assert(cTime>=_t);

    const dtype dTAcc = (dtype) __forwardIntDt;

    //Set up all ptrs and maps
    dtype xCPtr[dimTot]; //Get array
    Map<Matrix<dtype, dimTot, 1>> xC(xCPtr); //Get map object
    xC = _cStateCtrl;

    dtype vCPtr[dimTot]; //Get array
    Map<Matrix<dtype, dimTot, 1>> vC(vCPtr); //Get map object

    dtype cStateTempPtr[dimTot];
    Map<Matrix<dtype, dimTot, 1>> cStateTemp(cStateTempPtr); //Get map object
    Matrix<dtype, dimTot, 1> cStateTempM; //Arg

    dtype cVelTempPtr[dimTot];
    Map<Matrix<dtype, dimTot, 1>> cVelTemp(cVelTempPtr); //Get map object
    Matrix<dtype, dimTot, 1> cVelTempM; //Arg


    //Integrate till new timepoint
    simpleIntegratePtr<dimTot>((dtype) (cTime-_t), xCPtr, _fDyn, _fConv);
    //simpleIntegratePtr1<dimTot>((dtype) (cTime-_t), xCPtr, thisGetCtrlSpaceVelPtr);
    _cStateCtrl = xC; //Copy back
    //simpleIntegrate<dimTot>((dtype) (cTime-_t), _cStateCtrl, _fDyn, _fConv);
    //simpleIntegrate<dimTot, diffeoCtrl<dimTot, dimInt>>((dtype) (cTime-_t), _cStateCtrl, *this);
    // Set new internal point
    _t = cTime;

    //Keep demspace and control space consistent
    _cStateDem = _cStateCtrl;
    inverseTransform(_cStateDem); // -> ctrl 2 dem
    //Update the obtained point with new info from "reality"
    setPosDem(posMeas);

    //Get the velocity (ctrl space)
    //_fDyn(_cStateCtrl,_velTmpCtrl);
    _fDyn(xCPtr,vCPtr);
    //_fConv(_cStateCtrl, _velTmpCtrl);
    _fConv(xCPtr,vCPtr);
    //getCtrlSpaceVelPtr(xCPtr,vCPtr);

    // Copy the ctrl space velocity
    ctrlSpaceVel = vC;

    //(*this)(_cStateCtrl, _velTmpCtrl);

    //Continue forward integration to compute acceleration
    cStateTemp = _cStateCtrl;
    simpleIntegratePtr<dimTot>(dTAcc, cStateTempPtr, _fDyn, _fConv);
    //simpleIntegratePtr1<dimTot>(dTAcc, cStateTempPtr, thisGetCtrlSpaceVelPtr);
    //simpleIntegrate<dimTot>(dTAcc, cStateTemp, _fDyn, _fConv);
    //simpleIntegrate<dimTot, diffeoCtrl<dimTot, dimInt>>(dTAcc, cStateTemp, *this);

    //Get the velocity of continued point(ctrl space)
    //_fDyn(cStateTemp,cVelTemp);
    //_fConv(cStateTemp,cVelTemp);
    _fDyn(cStateTempPtr,cVelTempPtr);
    _fConv(cStateTempPtr,cVelTempPtr);
    //(*this)(cStateTemp,cVelTemp);
    //getCtrlSpaceVelPtr(cStateTempPtr,cVelTempPtr);

    // Compute ctrl space acceleration
    ctrlSpaceAcc = (cVelTempM-ctrlSpaceVel).array()/dTAcc;

    //Transform into demonstration space
    //demSpaceVel = _scalingInverse*( (_cStateJac*_velTmpCtrl).array() );
    //Using the mapped pointer
    demSpaceVel = _scalingInverse*( (_cStateJac*vC).array() );
    //Transform the forward integration point  // Forward transform -> dem 2 ctrl; Inverse transform ctrl 2 dem
    //Arg
    cStateTempM = cStateTemp;
    cVelTempM = cVelTemp;
    inverseTransformV(cStateTempM, cVelTempM);
    ///demSpaceVel = _velTmpCtrl;

    //Compute demonstration space acceleration
    demSpaceAcc = (cVelTempM-demSpaceVel).array()/dTAcc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Utility function
/*
 * Get the diffeo control class from the definition folder created by python
 * containing the dynamicsFile, diffeoFile, diffeoCtrlFile, convergenceFile
 */
/*
template <long dimTot, long dimInt>
diffeoCtrl<dimTot, dimInt> getDiffeoCtrlFromFolder(string definitionPath){

    assert(false && "TBD set function incorrect when copyconstructor is invoked");

    checkEndOfPath(definitionPath);

    cout << "Start loading definition files" << endl;
    gmm<dimTot, dimTot> aGMM(definitionPath+"dynamicsFile");
    diffeomorphism<dimTot> thisDiffeo(definitionPath+"diffeoFile");
    diffeoCtrl<dimTot,1> thisDiffeoCtrl(definitionPath+"diffeoCtrlFile");
    Matrix<dtype, dimTot+1, 1> convergenceDef = schlepil2::ReadMatrix(definitionPath+"convergenceFile");


    //Create the functions using pointers
    //Dynamics
    function<void(const dtype * const, dtype * const)> fDyn = bind(&gmm<dimTot,dimTot>::evalMap1Ptr, &aGMM,std::placeholders::_1, std::placeholders::_2);

    // Forced convergence
    function<void(const dtype * const, dtype * const)> fConv = bind(&ensureRadMinConv2Ptr<dimTot,true>, std::placeholders::_1, std::placeholders::_2, (dtype) convergenceDef(0,0),
                                                                    Matrix<dtype,dimTot,1>(convergenceDef.template block<dimTot,1>(1,0)));

    // Set
    thisDiffeoCtrl._diffeo = thisDiffeo;
    thisDiffeoCtrl._fDyn = fDyn;
    thisDiffeoCtrl._fConv = fConv;

    return thisDiffeoCtrl;
};
*/


#endif //CPP_DIFFEOCTRL_H
