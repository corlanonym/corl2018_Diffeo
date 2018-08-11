//
// Created by elfuius on 08/05/18.
//

#ifndef CPP_DIFFEOLIB_H
#define CPP_DIFFEOLIB_H

#include "combinedDiffeoCtrl.h"
#include "combinedDynamics.h"
//#include "gaussianKernel.h"
#include "localWeightDef.h"
#include "gmm.h"
#include "schlepilUtils.h"
#include "fileVector.h"

#include <map>

//Some typedefs for convenience since we are goig to mainly work in 2d
// Single transformations
typedef kernel<2> kernelTrans2d;
typedef kernel<3> kernelTrans3d;
// multitransformations
typedef multiTrans<2> multiTrans2d;
typedef multiTrans<3> multiTrans3d;
// Diffeo
typedef diffeomorphism<2> diffeo2d;
typedef diffeomorphism<3> diffeo3d;
// Combined to ctrl law
typedef combinedDiffeoCtrl<2,0> diffeoCtrl2d;
typedef combinedDiffeoCtrl<3,1> diffeoCtrl2dTimed;
template <long dim>
using diffeoCtrlDim = combinedDiffeoCtrl<dim,0>;
template <long dim>
using diffeoCtrlDimTimed = combinedDiffeoCtrl<dim,1>;

// Magnitude model
typedef gmm<2,1> magModel2d;
typedef gmm<3,1> magModel3d;
template <long dim>
using magModelDimed = gmm<dim,1>;

//Direction model
// One direction
typedef convergingDirections<2,1> direc2d;
typedef convergingDirections<2,0> direc2dFinal;
typedef convergingDirections<2,1> direc3d;
typedef convergingDirections<2,0> direc3dFinal;
template<long dim>
using direcDimed = convergingDirections<dim,1>;
template<long dim>
using direcFinalDimed = convergingDirections<dim,0>;
// The weighting kernels
//typedef weightKernel<2> weight2d; //The weights do not really have an independent variable
//typedef weightKernel<3> weight3d; //The weights do not really have an independent variable
//using weight2d = weightKernel<2>;
//using weight3d = weightKernel<3>;
// Combine to locally weighted
typedef locallyWeightedDirections<2, direc2d, weight2d> localDir2d;
typedef locallyWeightedDirections<2, direc2dFinal, weight2d> localDir2dFinal;
typedef locallyWeightedDirections<3, direc3d, weight3d> localDir3d;
typedef locallyWeightedDirections<3, direc3dFinal, weight3d> localDir3dFinal;
template<long dim>
using localDirDimed = locallyWeightedDirections<dim, direcDimed<dim>, weightKernel<dim>>;
template<long dim>
using localDirFinalDimed = locallyWeightedDirections<dim, direcFinalDimed<dim>, weightKernel<dim>>;
// Functors to ensure convergence
typedef minimallyConvergingDirection<2> convDirections2d;
typedef minimallyConvergingDirection<3> convDirections3d;

//Assmeble the control law
typedef combinedLocallyWeightedDirections<2, localDir2d, localDir2dFinal, convDirections2d> combinedDir2d;
typedef combinedLocallyWeightedDirections<3, localDir3d, localDir3dFinal, convDirections3d> combinedDir3d;
template<long dim>
using combinedDirDimed = combinedLocallyWeightedDirections<dim, localDirDimed<dim>, localDirFinalDimed<dim>, minimallyConvergingDirection<dim>>;




template<long dimTot, long dimInt>
combinedDiffeoCtrl<dimTot,dimInt> getDiffeoCtrl(string definitionFolder){

    typedef kernel<dimTot> thisKernel;
    typedef multiTrans<dimTot> thisMultiTrans;
    typedef diffeomorphism<dimTot> thisDiffeo;

    typedef combinedDiffeoCtrl<dimTot,dimInt> thisDiffeoCtrl;

    checkEndOfPath(definitionFolder);

    //Construct
    //construct the combinedControl object and the diffeo. The function pointers are constructed elsewhere
    thisDiffeo myDiffeo(definitionFolder+"diffeo.txt");

    thisDiffeoCtrl myDiffeoCtrl(definitionFolder+"combinedControl.txt");

    myDiffeoCtrl._diffeo = myDiffeo;
    myDiffeoCtrl.reset();

    return myDiffeoCtrl;
}

template <long dim>
magModelDimed<dim> getMagModel(string definitionFolder){

    checkEndOfPath(definitionFolder);

    magModelDimed<dim> myMagModel(definitionFolder+"magModel.txt");
    return myMagModel;
}

//template<long dim>
//combinedLocallyWeightedDirections<dim, locallyWeightedDirections<dim, convergingDirections<dim,1>, weightKernel<dim>>, locallyWeightedDirections<dim, convergingDirections<dim,0>,
//        weightKernel<dim>>, minimallyConvergingDirection<dim>>
template<long dim>
combinedDirDimed<dim> getDirModel(string definitionFolder){
    /*
    typedef convergingDirections<dim,1> thisDirec;
    typedef convergingDirections<dim,0> thisDirecFinal;
    typedef weightKernel<dim> thisWeight;
    typedef locallyWeightedDirections<dim, thisDirec, thisWeight> thisLocalDir;
    typedef locallyWeightedDirections<dim, thisDirecFinal, thisWeight> thisLocalDirFinal;
    typedef minimallyConvergingDirection<dim> thisConv;
    typedef combinedLocallyWeightedDirections<dim, thisLocalDir, thisLocalDirFinal, thisConv> thisCombinedDir;
     */
    checkEndOfPath(definitionFolder);

    combinedDirDimed<dim> myCombinedDir(definitionFolder+"dirModel.txt");

    return myCombinedDir;
}


#endif //CPP_DIFFEOLIB_H
