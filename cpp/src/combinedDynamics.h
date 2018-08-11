//
// Created by elfuius on 20/04/18.
//

#ifndef CPP_COMBINEDDYNAMICS_H
#define CPP_COMBINEDDYNAMICS_H

#include <Eigen/Core>
#include <algorithm>
#include <functional>
#include <cmath>
#include <iomanip>

#include "constants.h"
#include "myUtils.h"
#include "schlepilUtils.h"

using namespace Eigen;
using namespace std;

/*********************************************************************************************************************************************/
// Helper functions
/*********************************************************************************************************************************************/
template<long dim>
class minimallyConvergingDirection {
public:
    minimallyConvergingDirection(dtype minAng=0.){_minAng = minAng;};
    minimallyConvergingDirection(const string & defString){
                                                            try{
                                                                ifstream defFile(defString);
                                                                assert(defFile.is_open());
                                                                defFile >> _minAng;
                                                            }catch(...){
                                                                stringstream defStringStream(defString);
                                                                defStringStream >> _minAng;
                                                            }

    }

    void operator()(const Matrix<dtype, dim, 1> & xIn, Matrix<dtype, dim, 1> & dir){

        dtype coefFac = 1.1;
        dtype cosxDir;

        _x.noalias() = xIn/(xIn.norm() + dtype_eps);
        dir/=(dir.norm()+dtype_eps);

        cosxDir = _x.transpose()*dir + _minAng;
        while (cosxDir>0.){
            dir -= (coefFac*cosxDir)*_x;
            dir/=(dir.norm()+dtype_eps);
            cosxDir = (_x.transpose()*dir)(0,0) + _minAng;
            coefFac *=1.5;
        }
        if ((_x.transpose()*dir)(0,0) + _minAng>0.){
            cout << "Conv with " << endl << dir << endl << "at " << endl << xIn << endl;
            throw runtime_error("Ups");
        }
        dir/=(dir.norm()+dtype_eps);
    };

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

//private:
public:
    dtype _minAng;
    Matrix<dtype, dim, 1> _x;

};

/*********************************************************************************************************************************************/
// Converging directories
/*********************************************************************************************************************************************/
template <long dim, long dimE> // todo take proper care of dimE, currently only 0 and 1 is ok
class convergingDirections{
public:
    convergingDirections(); //sets all to zero
    convergingDirections(const string & defString);
    convergingDirections( const Matrix<dtype, dim,1> & x0, const Matrix<dtype, dim,1> & vp, const dtype alpha, const dtype beta );

    //No real destructor needed, no new
    ~convergingDirections(){};

    Matrix<dtype,dim,1> getDir(const Matrix<dtype, dim,1> & xIn);
    void getDir(const Matrix<dtype, dim,1> & xIn, Matrix<dtype, dim,1> & out);

    static long nPars(){return 2*dim+3;};

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

//private:
public: //debug

    void _compInternal();

    Matrix<dtype, dim,1> _x0;
    Matrix<dtype, dim,1> _vp;
    dtype _alpha;
    dtype _beta;

    Matrix<dtype, dim, dim-dimE> _Vnull; //Orthogonal directions
    Matrix<dtype, dim-dimE, dim> _Pnull; //nullspace projector for main-direction
    Matrix<dtype, dim, dim> _R; //Rotation matrix such that first dimension is along main velocity

    // temporary stuff
    Matrix<dtype, dim, 1> _xlocal; //local coords
    Matrix<dtype, dim-dimE, 1> _h; //orthogonal proj
    Matrix<dtype, dim-dimE, 1> _hsign;
    Matrix<dtype, dim-dimE, 1> _habs;
};

/*********************************************************************************************************************************************/
// locally weighted directions
/*********************************************************************************************************************************************/
template <long dim, typename dirType, typename weightType>
class locallyWeightedDirections{
public:
    locallyWeightedDirections(){
                                _dir = dirType();
                                _weight = weightType();
                                _relWeight=1.;};

    locallyWeightedDirections(const dirType & aDirection, const weightType & aWeight, dtype relWeight=1.){
                                                                                      _dir = aDirection;
                                                                                      _weight = aWeight;
                                                                                      _relWeight=relWeight;};
    locallyWeightedDirections(const string & defString);

    ~locallyWeightedDirections(){};

    static long nPars(){return dirType::nPars()+weightType::nPars()+2;};//addtional pars: Dimension; relWeight

    void getDir( const Matrix<dtype,dim,1> & xIn,  Matrix<dtype,dim,1> & out){
                                                                                _dir.getDir(xIn, out);
                                                                                out.array()*=(((dtype)_weight.getWeight1(xIn))*_relWeight);
                                                                             }; // Directions will get normed anyways
    Matrix<dtype,dim,1> getDir( const Matrix<dtype,dim,1> & xIn ){
                                                                  Matrix<dtype,dim,1> out;
                                                                  getDir(xIn, out);
                                                                  return out;
                                                                  };

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

//private:
public: //Debug
    dirType _dir;
    weightType _weight;
    dtype _relWeight;
};

/*********************************************************************************************************************************************/
// combined locally weighted directions
/*********************************************************************************************************************************************/
template <long dim, typename aLocalDir, typename aLocalDirFinal, typename aConvClass>
class combinedLocallyWeightedDirections{
public:
    combinedLocallyWeightedDirections();
    combinedLocallyWeightedDirections(const string & defString); // This will try to open the file if this failes it will assume a proper defstring
    ~combinedLocallyWeightedDirections(){_vecOfDir.clear();};

    void addLocallyWeightedDirection(const aLocalDir & newDir);

    void getDir(const Matrix<dtype, dim, 1> & xIn, Matrix<dtype, dim, 1> & out);
    Matrix<dtype, dim, 1> getDir(const Matrix<dtype, dim, 1> & xIn);

    void getDirTraj(const Matrix<dtype, dim, 1> & xIn, Matrix<dtype, 1, -1> & t, Matrix<dtype, dim, -1> & x, const dtype timeStep=2.5e-2);

    void setDynamicsFunction(){
        _fDyn = bind(static_cast<void(combinedLocallyWeightedDirections<dim,aLocalDir,aLocalDirFinal,aConvClass>::*)(const Matrix<dtype, dim, 1> &, Matrix<dtype, dim, 1> &)>(&combinedLocallyWeightedDirections<dim,aLocalDir,aLocalDirFinal,aConvClass>::getDir),
                     this, placeholders::_1, placeholders::_2);
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

//private:
public: //debug
    vector<aLocalDir, aligned_allocator<aLocalDir>> _vecOfDir;
    aConvClass _conv;
    dtype _baseConv;

    aLocalDirFinal _finalDir;
    bool _hasFinal;

    //Function
    function<void(const Matrix<dtype, dim,1> &, Matrix<dtype, dim,1>&)> _fDyn;
    
};

/*********************************************************************************************************************************************/
// Implementation convergingDirections
/*********************************************************************************************************************************************/
template <long dim, long dimE>
convergingDirections<dim, dimE>::convergingDirections(){
    _alpha = 0.;
    _beta = 0.;
    _x0.setZero();
    _vp.setZero();
    _Vnull.setZero();
    _Pnull.setZero();
    _R.setZero();
}
///////////////////////////////////////////////////////////////////////////
template <long dim, long dimE>
convergingDirections<dim, dimE>::convergingDirections(const string & defString){
    //Create from string
    Matrix<dtype, dim, 1> x0, vp;
    dtype alpha, beta, tmp;
    long tmpLong;

    istringstream defStringStream(defString);

    //Input is expected as dim;x0;vp;alpha;beta
    defStringStream >> tmpLong;
    if (not (tmpLong==dim)){
        cout << "Error at " << __LINE__ << endl;
        throw runtime_error("Inconsistent dimension - convergingDirections");
    }
    //position
    for (long i=0; i<dim; ++i){
        defStringStream >> tmp;
        x0(i,0) = tmp;
    }
    //direction
    for (long i=0; i<dim; ++i){
        defStringStream >> tmp;
        vp(i,0) = tmp;
    }
    //other
    defStringStream >> alpha;
    defStringStream >> beta;

    //Call init
    this->~convergingDirections();
    new (this) convergingDirections(x0, vp, alpha, beta);
}

///////////////////////////////////////////////////////////////////////////
template <long dim, long dimE>
convergingDirections<dim,dimE>::convergingDirections(const Matrix<dtype, dim, 1> & x0, const Matrix<dtype, dim, 1> & vp, const dtype alpha, const dtype beta) {
    //Easy stuff first
    _alpha = alpha;
    _beta = beta;

    _x0 = x0;
    _vp = vp;
    _compInternal();
}
/*********************************************************************************************************************************************/

template <long dim, long dimE>
void convergingDirections<dim,dimE>::_compInternal() {

    dtype vpNorm = _vp.norm();

    if (abs(vpNorm-1.)>1.e-8){
        _vp.array() /= (vpNorm+dtype_eps);
    }

    Matrix<dtype, -1, -1> VnullTmp = nullspace(Matrix<dtype, 1, dim>(_vp.transpose()));
    if ( (VnullTmp.cols()!=_Vnull.cols()) or (VnullTmp.rows()!=_Vnull.rows())){
        throw runtime_error("Nullspace not compatible!");
    }

    _Vnull = VnullTmp;
    _Pnull = _Vnull.transpose();
    _R.row(0) = _vp.transpose();
    _R.bottomRows(dim - dimE) = _Pnull;
}
/*********************************************************************************************************************************************/
template <long dim, long dimE>
void convergingDirections<dim, dimE>::getDir(const Matrix<dtype, dim, 1> &xIn, Matrix<dtype, dim, 1> &out) {

    _xlocal = xIn - _x0; //to local coords
    _h = _Pnull*_xlocal; //proj

    // Non-convergent zone
    if (_alpha>0.){
        _hsign = _h.array().sign();
        _habs = _h.array().abs();

        _habs.array() -= _alpha;
        // Replace all values smaller than 0 with zero
        _habs.array() = _habs.array().max(0.);

        _h.array() = _hsign.array()*_habs.array();
    }

    //scale
    _h.array() *= _beta;

    // add up vel and orthogonal vel
    out = _vp;
    out.array() += (_Vnull.array().rowwise()*(_h.transpose().array())).rowwise().sum();

    return;
}
///////////////////////////////////////////////////////////////////////////
template <long dim, long dimE>
Matrix<dtype, dim, 1> convergingDirections<dim, dimE>::getDir(const Matrix<dtype, dim, 1> &xIn) {
    Matrix<dtype, dim, 1> out;
    getDir(xIn, out);
    return out;
}

/*********************************************************************************************************************************************/
// Implementation locallyWeightedDirections
/*********************************************************************************************************************************************/
template <long dim, typename dirType, typename weightType>
locallyWeightedDirections<dim, dirType, weightType>::locallyWeightedDirections(const string & defString){

    //Split up the string and call constructor
    //String has to define direction than weight
    string defStringDir = "";
    string defStringWeight = "";
    string tmpString;
    const long nParsDir = dirType::nPars();
    const long nParsWeight = weightType::nPars();
    long tmpLong;
    dtype tmpWeight;

    istringstream defStringStream(defString);

    defStringStream >> tmpLong;
    if (not (tmpLong==dim)){
        cout << "defString is " << endl << defString << endl << ".";
        cout << "Error at " << __LINE__ << endl;
        throw runtime_error("Inconsistent dimension - locallyWeighted");
    };
    defStringStream >> tmpWeight;

    //Offstream leaves empty line
    getline(defStringStream, tmpString);//Dummy read

    for (long i=0; i<nParsDir; i++){
        getline(defStringStream, tmpString);
        defStringDir += tmpString;
        defStringDir += "\n";
    }

    for (long i=0; i<nParsWeight; i++){
        getline(defStringStream, tmpString);
        defStringWeight += tmpString;
        defStringWeight += "\n";
    }

    //Construct on the fly and call constructor
    //dirType thisDir(defStringDir);
    //weightType thisWeight(defStringWeight);
    //cout << "a" << endl;
    //locallyWeightedDirections(thisDir, thisWeight);
    this->~locallyWeightedDirections();
    new (this) locallyWeightedDirections(dirType(defStringDir), weightType(defStringWeight), tmpWeight);
};

/*********************************************************************************************************************************************/
// combinedLocallyWeightedDirections
/*********************************************************************************************************************************************/
template <long dim, typename aLocalDir, typename aLocalDirFinal, typename aConvClass>
combinedLocallyWeightedDirections<dim, aLocalDir, aLocalDirFinal, aConvClass>::combinedLocallyWeightedDirections(){
    _baseConv=(dtype) 0.;
    _vecOfDir.clear();
    _hasFinal = false;
};
template <long dim, typename aLocalDir, typename aLocalDirFinal, typename aConvClass>
combinedLocallyWeightedDirections<dim, aLocalDir, aLocalDirFinal, aConvClass>::combinedLocallyWeightedDirections(const string & defString) {

    //Reset
    _vecOfDir.clear();

    string actualDefString = "";
    string partialString, tmpString;
    const long nParsLocalDir = aLocalDir::nPars();
    long nComponents, dimCheck;

    try{
        //Interpret as file
        std::ifstream defFile(defString);
        //Read to string
        actualDefString.assign( (istreambuf_iterator<char>(defFile) ), (istreambuf_iterator<char>()) );
    }catch(...){
        //Copy the string
        actualDefString = defString;
    }

    //Clean up
    this->~combinedLocallyWeightedDirections();
    new (this) combinedLocallyWeightedDirections();

    // Excpected string style
    // dim
    // baseConv
    // nComponents
    // [for each localDir]
    // nParsLocalLines

    istringstream defStringStream(actualDefString);

    defStringStream >> dimCheck;
    assert((dimCheck==dim) && "Incompatible definition of dim");

    //Get the baseconv
    defStringStream >> _baseConv;

    defStringStream >> nComponents;
    assert((nComponents>=0) && "Has to be non-negative");

    //Offstream leaves empty line
    getline(defStringStream, tmpString);//DummyRead

    for (long i=0; i<nComponents; ++i){
        partialString = "";
        for (long j=0; j<nParsLocalDir; ++j){
            getline(defStringStream, tmpString);
            partialString += tmpString;
            partialString += "\n";
        }
        //push_back
        if (i==nComponents-1){
            cout << "last" << endl;
            //Check which type the last one is; Regular constructor will fail if final
            try{
                _vecOfDir.push_back( aLocalDir(partialString) );
                _hasFinal=false;
            }catch(...){
                _finalDir = aLocalDirFinal(partialString);
                _hasFinal = true;
            }
        }else{
            //Regular converging directions
            _vecOfDir.push_back( aLocalDir(partialString) );
        }
    }
    //Done
}

/*********************************************************************************************************************************************/
template <long dim, typename aLocalDir, typename aLocalDirFinal, typename aConvClass>
void combinedLocallyWeightedDirections<dim, aLocalDir, aLocalDirFinal, aConvClass>::addLocallyWeightedDirection(const aLocalDir & newDir){
    _vecOfDir.push_back(newDir);
};

/*********************************************************************************************************************************************/
template <long dim, typename aLocalDir, typename aLocalDirFinal, typename aConvClass>
void combinedLocallyWeightedDirections<dim, aLocalDir, aLocalDirFinal, aConvClass>::getDir(const Matrix<dtype, dim, 1> &xIn, Matrix<dtype, dim, 1> &out) {

    dtype xnorm = xIn.norm();

    out = (_baseConv/(xnorm+dtype_eps))*xIn;

    //Loop to sum up all weighted directions
    for (typename vector<aLocalDir, aligned_allocator<aLocalDir>>::iterator it = _vecOfDir.begin(); it != _vecOfDir.end(); ++it){
        out += it->getDir(xIn);
    }

    if(_hasFinal){
        out += _finalDir.getDir(xIn);
    }

    //Normalize
    out /= (out.norm()+dtype_eps);

    //Ensure geometric convergence
    _conv(xIn, out);

    //Tends to zero when approaching zero
    out *= min(xnorm*1.e4, 1.);

    //assert((xIn/(xnorm+dtype_eps)).transpose()*out < -0.001);
}
////////////////////////////////////////////////////////////////////////////////////////
template <long dim, typename aLocalDir, typename aLocalDirFinal, typename aConvClass>
Matrix<dtype, dim, 1> combinedLocallyWeightedDirections<dim, aLocalDir, aLocalDirFinal, aConvClass>::getDir(const Matrix<dtype, dim, 1> &xIn) {
    Matrix<dtype, dim, 1> out;
    getDir(xIn, out);
    return out;
}
template <long dim, typename aLocalDir, typename aLocalDirFinal, typename aConvClass>
void combinedLocallyWeightedDirections<dim, aLocalDir, aLocalDirFinal, aConvClass>::getDirTraj(const Matrix<dtype, dim, 1> &xIn, Matrix<dtype, 1, -1> &t, Matrix<dtype, dim, -1> &x, const dtype timeStep) {

    //Define some meta pars
    const long bufferSize = 1000;
    const long countNr = 200;

    //counter
    long counter = 0;

    //Init matrices
    t.resize(NoChange, bufferSize);
    x.resize(NoChange, bufferSize);
    t(0,0)=0.;
    x.col(0) = xIn;

    //void simpleIntegrate(const dtype tEnd, Matrix<dtype, dim,1> & x, function<void(const Matrix<dtype, dim,1> &, Matrix<dtype, dim,1>&)> & fDyn, const dtype dt=5.e-4)

    Matrix<dtype, dim, 1> xC = xIn;

    while (xC.squaredNorm()>(ctrlSpaceAssumeFinishedRadius_*ctrlSpaceAssumeFinishedRadius_)){
        //Increment
        counter++;
        if ((counter%countNr)==0) {
            cout << "current point is " << endl << xC << endl << " with norm " << xC.norm() << endl;
        }

        //Check buffer size
        if(counter>=x.cols()){
            t.conservativeResize(NoChange, x.cols()+bufferSize);
            x.conservativeResize(NoChange, x.cols()+bufferSize);
        }

        //step
        simpleIntegrate<dim>(timeStep, xC, _fDyn);

        //save
        t(0,counter) = counter*timeStep;
        x.col(counter) = xC;
    }

    t.conservativeResize(NoChange, counter+2);
    x.conservativeResize(NoChange, counter+2);

    t(0,counter+1) = (counter+1)*timeStep;
    x.col(counter+1).setZero();
}

#endif //CPP_COMBINEDDYNAMICS_H
