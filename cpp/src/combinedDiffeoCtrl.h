//
// Created by elfuius on 11/02/18.
//

#ifndef CPP_COMBINEDDIFFEOCTRL_H
#define CPP_COMBINEDDIFFEOCTRL_H

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


using namespace Eigen;
using namespace std;

//We consider autonous systems with or without internal variables
template <long dimTot, long dimInt>
class combinedDiffeoCtrl{
public:
    //combinedDiffeoCtrl(){_t=0.;_scaling.setOnes();_scalingInverse.setOnes(); _demSpaceOffset.setZero(); thisGetCtrlSpaceVelPtr = std::bind(&combinedDiffeoCtrl<dimTot, dimInt>::getCtrlSpaceVelPtr, this, std::placeholders::_1, std::placeholders::_2);};
    combinedDiffeoCtrl(){_t=0.;_scaling.setOnes();_scalingInverse.setOnes(); _demSpaceOffset.setZero(); _minMagVel=1e-3; setDynamicsFunction();};
    combinedDiffeoCtrl(const string & fileName);
    //No new so nothing to do
    ~combinedDiffeoCtrl(){};


    //Reset external time
    void reset(){_t=0.;};

    //Check if the given demonstration point is close to the target
    bool isFinished(const Matrix<dtype,dimTot-dimInt,1> & xCurrent, const dtype epsNorm=1e-2);

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
    void setPosDemTot(const Matrix<dtype, dimTot,1> & newPos){ _cStateDem = newPos; _cStateCtrl=_cStateDem;
                                                               _cStateJac.setIdentity();
                                                               forwardTransformJac(_cStateCtrl, _cStateJac, 0,-1,true,'r');};
    //Set the non-internal part in demonstration
    void setPosDemExt(const Matrix<dtype, dimTot-dimInt,1> & newPos){  _cStateDem.template block<dimTot-dimInt,1>(0,0) = newPos; _cStateCtrl=_cStateDem;
                                                                       _cStateJac.setIdentity();
                                                                       forwardTransformJac(_cStateCtrl,_cStateJac,0,-1,true,'r');};
    // Set all in ctrl
    // Inverse transform goes from ctrl 2 dem, so no jacobian inversion needed
    void setPosCtrlTot(const Matrix<dtype, dimTot,1> & newPos){_cStateCtrl = newPos; _cStateDem=_cStateCtrl;
                                                               _cStateJac.setIdentity();
                                                               inverseTransformJac(_cStateDem, _cStateJac,0,-1,false,'l');};
    // Set non-internal in ctrl
    void setPosCtrlExt(const Matrix<dtype, dimTot-dimInt,1> & newPos){_cStateCtrl.template block<dimTot-dimInt,1>(0,0) = newPos; _cStateDem=_cStateCtrl;
                                                                      _cStateJac.setIdentity();
                                                                      inverseTransformJac(_cStateDem, _cStateJac,0,-1,false,'l');};

    //The same as above as reference getters
    template<typename thisDType>
    void getPosDemExt(Matrix<thisDType, dimTot-dimInt, 1> & out)const{out = _cStateDem.template block<dimTot-dimInt, 1>(0,0).template cast<thisDType>();};
    template<typename thisDType>
    void getPosDemTot(Matrix<thisDType, dimTot, 1> & out)const{out = _cStateDem.template cast<thisDType>();};
    template<typename thisDType>
    void getPosCtrlExt(Matrix<thisDType, dimTot-dimInt, 1> & out)const{out = _cStateCtrl.template block<dimTot-dimInt, 1>(0,0).template cast<thisDType>();};
    template<typename thisDType>
    void getPosCtrlTot(Matrix<thisDType, dimTot, 1> & out)const{out = _cStateCtrl.template cast<thisDType>();};

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

    //Evaluate the dynamics
    //Attention x and v will be modified
    void getDemSpaceVel( Matrix<dtype,dimTot,1> & x, Matrix<dtype,dimTot,1> & v );

    void setDynamicsFunction() {
                                //todo check if this has to be reset if function are reset
                                _fDyn = bind(&combinedDiffeoCtrl<dimTot, dimInt>::getDemSpaceVel, this, placeholders::_1, placeholders::_2 );
                              };


    //void getCtrlSpaceVelPtr(const dtype * const x, dtype * const v){ this->_fDyn(x,v); this->_fConv(x,v);};



    //Necessary function
    // Eval dynamic in ctrl space
    // First argument is the point cosidered in control space, second argument is where to store velocity
    // std::function is not the way to go for performance
    // However alternatives seem to be annoying ->

    function<void(const Matrix<dtype, dimTot,1> &, Matrix<dtype, dimTot,1> &)> _fDir; //Takes the control space point and outputs control space direction
    function<void(const Matrix<dtype, dimTot,1> &, Matrix<dtype, 1,1> &)> _fMag; //Takes demonstration space point and outputs magnitude

    function<void(Matrix<dtype,dimTot,1> &, Matrix<dtype,dimTot,1> &)> _fDyn; //Actual demonstration space dynamics

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

    dtype _minMagVel;
};


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// IMPLEMENTATIONS COMBINEDDIFFEOCTRL
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <long dimTot, long dimInt>
combinedDiffeoCtrl<dimTot, dimInt>::combinedDiffeoCtrl(const string & fileName):combinedDiffeoCtrl() {


    //Clean up
    this->~combinedDiffeoCtrl();
    new (this) combinedDiffeoCtrl();

    string cline;
    string::size_type sz;
    dtype tmpVal;
    long tmpLong;
    bool tmpBool;

    Matrix<dtype,dimTot,1> tmpScale;

    stringstream defStringStream;

    try {
        ifstream defFile(fileName);
        //assert(defFile.is_open() && "Could not open file");
        if (!defFile.is_open()){
            throw runtime_error("file not found -> Use as defstring");
        }
        defStringStream << defFile.rdbuf();
    }catch(...){
        defStringStream = stringstream(fileName);
    }

    //dimTot
    //dimInt
    //scaling
    defStringStream >> tmpLong;
    assert(tmpLong==dimTot);
    defStringStream >> tmpLong;
    assert(tmpLong==dimInt);
    defStringStream >> tmpBool;
    _direction = tmpBool;
    cout << "Direction set to " << _direction << endl;

    for (size_t i=0; i<dimTot; ++i){
        defStringStream >> tmpVal;
        tmpScale(i) = tmpVal;
    }
    setScaling(tmpScale);
    //Get the offset
    for (size_t i=0; i<dimTot; ++i){
        defStringStream >> tmpVal;
        _demSpaceOffset(i) = tmpVal;
    }
    setDynamicsFunction();
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <long dimTot, long dimInt>
bool combinedDiffeoCtrl<dimTot, dimInt>::isFinished(const Matrix<dtype, dimTot-dimInt, 1> & xCurrent, const dtype epsNorm) {

    Matrix<dtype, dimTot-dimInt, 1> xx = xCurrent-_demSpaceOffset.template topRows<dimTot-dimInt>();
    // We do not care about internal state for now

    return (xx.squaredNorm() < epsNorm*epsNorm); //Multiplication is faster than root calc
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <long dimTot, long dimInt>
void combinedDiffeoCtrl<dimTot, dimInt>::forwardTransform(Matrix<dtype, dimTot, 1> &xIn, const long kStart, const long kStop) {
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
void combinedDiffeoCtrl<dimTot, dimInt>::forwardTransformJac(Matrix<dtype, dimTot, 1> &xIn, Matrix<dtype, dimTot, dimTot> &Jac, long kStart, long kStop, const bool outInvJac,
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
void combinedDiffeoCtrl<dimTot, dimInt>::forwardTransformV(Matrix<dtype, dimTot, 1> &xIn, Matrix<dtype, dimTot, 1> &vIn, long kStart, long kStop) {

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
void combinedDiffeoCtrl<dimTot, dimInt>::inverseTransform(Matrix<dtype, dimTot, 1> &xInPrime, const long kStart, const long kStop) {
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
void combinedDiffeoCtrl<dimTot, dimInt>::inverseTransformJac(Matrix<dtype, dimTot, 1> &xInPrime, Matrix<dtype, dimTot, dimTot> &JacPrime, const long kStart, const long kStop, const bool outInvJac,
                                                    const char whichSide) {
    /*
     * Inverse transform: From ctrl to dem
     */
    // Attention jacobians within combinedDiffeoCtrl are always from ctrl to demonstration
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
void combinedDiffeoCtrl<dimTot, dimInt>::inverseTransformV(Matrix<dtype, dimTot, 1> &xInPrime, Matrix<dtype, dimTot, 1> &vInPrime, const long kStart, const long kStop) {
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
Matrix<dtype, dimTot, -1> combinedDiffeoCtrl<dimTot, dimInt>::getTraj(const Matrix<dtype, 1, -1> & tVec, bool outInDem) {

    /*
     * Compute forward trajectory for current time and position; will leave the object unchanged
     */

    Matrix<dtype, dimTot, -1> xOut = Matrix<dtype, dimTot, -1>::Zero(dimTot, tVec.cols());

    Matrix<dtype, dimTot, 1> xC = _cStateDem;
    Matrix<dtype, dimTot, 1> xtmp;

    dtype tTraj = _t;

    assert(tVec[0]>=tTraj);
    cout << "integrating for " << tVec.cols() << " steps" << endl;
    //cout << tVec << endl;
    for (size_t i=0; i<tVec.cols(); ++i){
        //cout << "step " << i << endl;
        simpleIntegrate<dimTot>((dtype) tVec[i]-tTraj, xC, _fDyn);
        if (outInDem){
            xOut.col(i) = xC;
        }else{
            //Transform from dem 2 ctrl
            xtmp = xC;
            forwardTransform(xtmp);
            xOut.col(i) = xtmp;
        }
        tTraj = tVec[i];
    }
    return  xOut;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <long dimTot, long dimInt>
void combinedDiffeoCtrl<dimTot, dimInt>::getNextVel(const Matrix<dtype, dimTot - dimInt, 1> &posMeas, const dtype cTime, Matrix<dtype,dimTot,1>& demSpaceVel) {

    if (dimInt == 0){
        // The dynamics do not have an internal state -> They are a vector field of the measured demonstration space
        Matrix<dtype, dimTot, 1> x = posMeas;
        getDemSpaceVel(x, demSpaceVel);
    }else{
        //The dynamics do have at least one internal state -> The last position has to be forward integrated and updated
        Matrix<dtype, dimTot, 1> dummyX = _cStateDem;
        assert(cTime>=_t);

        //Integrate till new timepoint
        simpleIntegrate<dimTot>((dtype) (cTime-_t), _cStateDem, _fDyn); //Integrate in demonstration space -> speed-up update

        //Update
        setPosDemExt(posMeas);

        //Get the new velocity
        getDemSpaceVel(dummyX, demSpaceVel);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <long dimTot, long dimInt>
void combinedDiffeoCtrl<dimTot, dimInt>::getNextVelNAcc(const Matrix<dtype, dimTot - dimInt, 1> &posMeas, const dtype cTime, Matrix<dtype,dimTot,1>& demSpaceVel, Matrix<dtype,dimTot,1>& demSpaceAcc) {
    assert(cTime>=_t);

    const dtype dTAcc = (dtype) __forwardIntDt;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// void getNextVelNAcc(const Matrix<dtype,dimTot-dimInt,1> & posMeas, const dtype cTime, Matrix<dtype,dimTot,1>& demSpaceVel, Matrix<dtype,dimTot,1>& demSpaceAcc, Matrix<dtype,dimTot,1>& ctrlSpaceVel, Matrix<dtype,dimTot,1>& ctrlSpaceAcc);
template <long dimTot, long dimInt>
void combinedDiffeoCtrl<dimTot, dimInt>::getNextVelNAcc(const Matrix<dtype, dimTot - dimInt, 1> &posMeas, const dtype cTime, Matrix<dtype,dimTot,1>& demSpaceVel, Matrix<dtype,dimTot,1>& demSpaceAcc, Matrix<dtype,dimTot,1>& ctrlSpaceVel, Matrix<dtype,dimTot,1>& ctrlSpaceAcc) {
    assert(cTime>=_t);

    const dtype dTAcc = (dtype) __forwardIntDt;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <long dimTot, long dimInt>
void combinedDiffeoCtrl<dimTot, dimInt>::getDemSpaceVel(Matrix<dtype, dimTot, 1> &x, Matrix<dtype, dimTot, 1> &v) {
    // Take a demonstration space point and evaluates the dynamics

    //First get magnitude
    //todo check for other possibility
    Matrix<dtype, 1,1 > thisMagnitudeM;
    _fMag(x, thisMagnitudeM);

    //Ensure that magnitude is positive (otherwise direction is inversed)
    dtype thisMagnitude = max(thisMagnitudeM(0,0), _minMagVel);

    //Transform the point into control space and get the jacobian
    _cStateJac.setIdentity();
    forwardTransformJac(x,_cStateJac,0,-1,true,'r'); //The jacobian from control to demonstration space is stored in _cStateJac now, x is now in control space

    //Reduce the velocity if close close to control space origin
    thisMagnitude = min(thisMagnitude, x.norm()*ctrlSpaceOriginFactor_);

    //Get the direction
    _fDir(x, v); //Control space direction in v

    //Test
    if (((x.transpose()*v)*thisMagnitude)(0,0)>0.){
        cout << "shit " << endl << "x" << endl << x << endl << "v" << v << endl << "mag " << thisMagnitude << "res " << ((x.transpose()*v)*thisMagnitude)(0,0) << endl;
        assert(false);
    }


    //Demonstration space direction
    v = _cStateJac*v;
    //Take care of scaling
    v.array() *= _scalingInverse;
    //Renormalize and apply magnitude
    v *= (thisMagnitude/(v.norm()+dtype_eps));
    //
    //Done
}



#endif //CPP_COMBINEDDIFFEOCTRL_H
