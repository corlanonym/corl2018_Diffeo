//
// Created by elfuius on 15/05/18.
//

#include <diffeoLib.h> //Has to be updated manually
#include <functional>
#include <string>

using namespace std;

template<long dimTot, long dimInt>
void forwardDimed(string definitionFolder, string inputFolder, string outputFolder){

    checkEndOfPath(definitionFolder);
    checkEndOfPath(inputFolder);
    checkEndOfPath(outputFolder);

    combinedDiffeoCtrl<dimTot,dimInt> myDiffeoControl = getDiffeoCtrl<dimTot, dimInt>(definitionFolder);

    //tmp
    Matrix<dtype, dimTot, 1> xtmp;
    //Read
    Matrix<dtype, dimTot, -1> xAll = ReadMatrix(inputFolder+"pos.txt");

    for (long i=0; i<xAll.cols(); ++i){
        xtmp = xAll.col(i);
        myDiffeoControl.forwardTransform(xtmp);
        xAll.col(i)=xtmp;
    }

    WriteMatrix(outputFolder+"pos.txt", xAll);
};

template<long dimTot, long dimInt>
void forwardVDimed(string definitionFolder, string inputFolder, string outputFolder){

    checkEndOfPath(definitionFolder);
    checkEndOfPath(inputFolder);
    checkEndOfPath(outputFolder);

    combinedDiffeoCtrl<dimTot,dimInt> myDiffeoControl = getDiffeoCtrl<dimTot, dimInt>(definitionFolder);

    //tmp
    Matrix<dtype, dimTot, 1> xtmp;
    Matrix<dtype, dimTot, 1> vtmp;
    //Read
    Matrix<dtype, dimTot, -1> xAll = ReadMatrix(inputFolder+"pos.txt");
    Matrix<dtype, dimTot, -1> vAll = ReadMatrix(inputFolder+"vel.txt");

    for (long i=0; i<xAll.cols(); ++i){
        xtmp = xAll.col(i);
        vtmp = vAll.col(i);
        myDiffeoControl.forwardTransformV(xtmp, vtmp);
        xAll.col(i)=xtmp;
        vAll.col(i)=vtmp;
    }

    WriteMatrix(outputFolder+"pos.txt", xAll);
    WriteMatrix(outputFolder+"vel.txt", vAll);
};

template<long dimTot, long dimInt>
void inverseDimed(string definitionFolder, string inputFolder, string outputFolder){

    checkEndOfPath(definitionFolder);
    checkEndOfPath(inputFolder);
    checkEndOfPath(outputFolder);

    combinedDiffeoCtrl<dimTot,dimInt> myDiffeoControl = getDiffeoCtrl<dimTot, dimInt>(definitionFolder);

    //tmp
    Matrix<dtype, dimTot, 1> xtmp;
    //Read
    Matrix<dtype, dimTot, -1> xAll = ReadMatrix(inputFolder+"pos.txt");

    for (long i=0; i<xAll.cols(); ++i){
        xtmp = xAll.col(i);
        myDiffeoControl.inverseTransform(xtmp);
        xAll.col(i)=xtmp;
    }

    WriteMatrix(outputFolder+"pos.txt", xAll);
};

template<long dimTot, long dimInt>
void inverseVDimed(string definitionFolder, string inputFolder, string outputFolder){

    checkEndOfPath(definitionFolder);
    checkEndOfPath(inputFolder);
    checkEndOfPath(outputFolder);

    combinedDiffeoCtrl<dimTot,dimInt> myDiffeoControl = getDiffeoCtrl<dimTot, dimInt>(definitionFolder);

    //tmp
    Matrix<dtype, dimTot, 1> xtmp;
    Matrix<dtype, dimTot, 1> vtmp;
    //Read
    Matrix<dtype, dimTot, -1> xAll = ReadMatrix(inputFolder+"pos.txt");
    Matrix<dtype, dimTot, -1> vAll = ReadMatrix(inputFolder+"vel.txt");

    for (long i=0; i<xAll.cols(); ++i){
        xtmp = xAll.col(i);
        vtmp = vAll.col(i);
        myDiffeoControl.inverseTransformV(xtmp, vtmp);
        xAll.col(i)=xtmp;
        vAll.col(i)=vtmp;
    }

    WriteMatrix(outputFolder+"pos.txt", xAll);
    WriteMatrix(outputFolder+"vel.txt", vAll);
};


template<long dimTot, long dimInt>
void demSpaceVelDimed(string definitionFolder, string inputFolder, string outputFolder, string demOrCtrl){

    //typedef combinedLocallyWeightedDirections<dimTot, locallyWeightedDirections<dimTot, convergingDirections<dimTot,1>, weightKernel<dimTot>>, locallyWeightedDirections<dimTot,
    //        convergingDirections<dimTot,0>, weightKernel<dimTot>>,
    //        minimallyConvergingDirection<dimTot>> thisCombinedDir;
    typedef combinedDirDimed<dimTot> thisCombinedDir;

    checkEndOfPath(definitionFolder);
    checkEndOfPath(inputFolder);
    checkEndOfPath(outputFolder);

    cout << "loading from " << definitionFolder << endl;

    combinedDiffeoCtrl<dimTot,dimInt> myDiffeoControl = getDiffeoCtrl<dimTot, dimInt>(definitionFolder);
    gmm<dimTot,1> myMagModel = getMagModel<dimTot>(definitionFolder);
    thisCombinedDir myDirModel = getDirModel<dimTot>(definitionFolder);
    minimallyConvergingDirection<dimTot> myConv(definitionFolder+"minConv.txt");

    //Set conv-functor
    myDirModel._conv = myConv;

    // Get the function pointers
    //Direction
    function<void(const Matrix<dtype, dimTot, 1> &, Matrix<dtype, dimTot, 1> &)> funcDir = bind(static_cast<void(thisCombinedDir::*)(const Matrix<dtype, dimTot, 1> &, Matrix<dtype, dimTot, 1> &)>(&thisCombinedDir::getDir), &myDirModel, placeholders::_1, placeholders::_2);
    //Magnitude
    function<void(const Matrix<dtype, dimTot, 1> &, Matrix<dtype, 1, 1> &)> funcMag = bind(static_cast<void(gmm<dimTot,1>::*)(const Matrix<dtype, dimTot, 1> &, Matrix<dtype, 1, 1> &)>(&gmm<dimTot,1>::evalMap1), &myMagModel, placeholders::_1, placeholders::_2);
    //Set
    myDiffeoControl._fDir = funcDir;
    myDiffeoControl._fMag = funcMag;

    //Check if in demonstration space
    assert( ((demOrCtrl == "dem") || (demOrCtrl == "ctrl")) && "Expected dem or ctrl" );
    if (demOrCtrl == "ctrl"){
        inverseDimed<dimTot,dimInt>(definitionFolder, inputFolder, inputFolder);
    }

    //Get the input
    Matrix<dtype, dimTot, -1> xAll = ReadMatrix(inputFolder+"pos.txt");
    Matrix<dtype, dimTot, -1> vAll(dimTot, xAll.cols());
    Matrix<dtype, dimTot, 1> xtmp, vtmp;

    for (long i=0; i<xAll.cols();++i){
        xtmp = xAll.col(i);
        myDiffeoControl.getDemSpaceVel(xtmp, vtmp);
        xAll.col(i) = xtmp;
        vAll.col(i) = vtmp;
    }

    cout << "saving pos to "<< outputFolder<<"pos.txt"<<endl;
    WriteMatrix(outputFolder+"pos.txt", xAll);
    cout << "saving vel to "<< outputFolder<<"vel.txt"<<endl;
    WriteMatrix(outputFolder+"vel.txt", vAll);
};

template<long dimTot, long dimInt>
void ctrlSpaceVelDimed(string definitionFolder, string inputFolder, string outputFolder, string demOrCtrl){

    //typedef combinedLocallyWeightedDirections<dimTot, locallyWeightedDirections<dimTot, convergingDirections<dimTot,1>, gaussianKernel<dimTot - 1, 1>>, locallyWeightedDirections<dimTot,
    //        convergingDirections<dimTot,0>, gaussianKernel<dimTot - 1, 1>>, minimallyConvergingDirection<dimTot>>
    //        thisCombinedDir;
    typedef combinedDirDimed<dimTot> thisCombinedDir;

    checkEndOfPath(definitionFolder);
    checkEndOfPath(inputFolder);
    checkEndOfPath(outputFolder);

    combinedDiffeoCtrl<dimTot,dimInt> myDiffeoControl = getDiffeoCtrl<dimTot, dimInt>(definitionFolder);
    thisCombinedDir myDirModel = getDirModel<dimTot>(definitionFolder);
    minimallyConvergingDirection<dimTot> myConv(definitionFolder+"minConv.txt");

    //Set conv-functor
    myDirModel._conv = myConv;

    // Get the function pointers
    //Direction
    function<void(const Matrix<dtype, dimTot, 1> &, Matrix<dtype, dimTot, 1> &)> funcDir = bind(static_cast<void(thisCombinedDir::*)(const Matrix<dtype, dimTot, 1> &, Matrix<dtype, dimTot, 1> &)>(&thisCombinedDir::getDir), &myDirModel, placeholders::_1, placeholders::_2);

    //Check if in demonstration space
    assert( ((demOrCtrl == "dem") || (demOrCtrl == "ctrl")) && "Expected dem or ctrl" );
    if (demOrCtrl == "dem"){
        forwardDimed<dimTot,dimInt>(definitionFolder, inputFolder, inputFolder);
    }

    //Get the input
    Matrix<dtype, dimTot, -1> xAll = ReadMatrix(inputFolder+"pos.txt");
    Matrix<dtype, dimTot, -1> vAll(dimTot, xAll.cols());
    Matrix<dtype, dimTot, 1> xtmp, vtmp;

    for (long i=0; i<xAll.cols();++i){
        xtmp = xAll.col(i);
        funcDir(xtmp, vtmp);
        vAll.col(i) = vtmp;
    }

    WriteMatrix(outputFolder+"vel.txt", vAll);
};

template<long dimTot, long dimInt>
void demSpaceTrajDimed(string definitionFolder, string inputFolder, string outputFolder, string demOrCtrl, bool outInDem){

    //typedef combinedLocallyWeightedDirections<dimTot, locallyWeightedDirections<dimTot, convergingDirections<dimTot,1>, gaussianKernel<dimTot - 1, 1>>, locallyWeightedDirections<dimTot,
    //       convergingDirections<dimTot,0>, gaussianKernel<dimTot - 1, 1>>, minimallyConvergingDirection<dimTot>>
    //        thisCombinedDir;
    typedef combinedDirDimed<dimTot> thisCombinedDir;

    checkEndOfPath(definitionFolder);
    checkEndOfPath(inputFolder);
    checkEndOfPath(outputFolder);

    combinedDiffeoCtrl<dimTot,dimInt> myDiffeoControl = getDiffeoCtrl<dimTot, dimInt>(definitionFolder);
    //gmm<dimTot,1> myMagModel = getMagModel<dimTot>(definitionFolder);
    magModelDimed<dimTot> myMagModel = getMagModel<dimTot>(definitionFolder);
    thisCombinedDir myDirModel = getDirModel<dimTot>(definitionFolder);
    minimallyConvergingDirection<dimTot> myConv(definitionFolder+"minConv.txt");

    //Set conv-functor
    myDirModel._conv = myConv;

    // Get the function pointers
    //Direction
    function<void(const Matrix<dtype, dimTot, 1> &, Matrix<dtype, dimTot, 1> &)> funcDir = bind(static_cast<void(thisCombinedDir::*)(const Matrix<dtype, dimTot, 1> &, Matrix<dtype, dimTot, 1> &)>(&thisCombinedDir::getDir), &myDirModel, placeholders::_1, placeholders::_2);
    //Magnitude
    function<void(const Matrix<dtype, dimTot, 1> &, Matrix<dtype, 1, 1> &)> funcMag = bind(static_cast<void(gmm<dimTot,1>::*)(const Matrix<dtype, dimTot, 1> &, Matrix<dtype, 1, 1> &)>(&gmm<dimTot,1>::evalMap1), &myMagModel, placeholders::_1, placeholders::_2);
    //Set
    myDiffeoControl._fDir = funcDir;
    myDiffeoControl._fMag = funcMag;

    //Get the input
    Matrix<dtype, dimTot, 1> xInit = ReadMatrix(inputFolder+"pos.txt");

    //Check if in demonstration space
    assert( ((demOrCtrl == "dem") || (demOrCtrl == "ctrl")) && "Expected dem or ctrl" );
    if (demOrCtrl == "ctrl"){
        myDiffeoControl.setPosCtrlTot(xInit);
    }else{
        myDiffeoControl.setPosDemTot(xInit);
    }

    try{
        //time vector is given -> return for these values
        cout << "computing for time vector" << endl;
        Matrix<dtype, 1, -1> tVec = ReadVector(inputFolder+"t.txt");
        Matrix<dtype, dimTot, -1> xAll = myDiffeoControl.getTraj(tVec, outInDem);
        cout << "Finished computing; resulting traj is of shape " << xAll.rows() << " ; " << xAll.cols() << " with" << endl;
        cout << xAll << endl;
        WriteMatrix(outputFolder+"traj.txt", xAll);
    }catch(...){
        //No t vector given
        cout << "computing till converged" << endl;
        long counter = 0;
        const long bufferLength = 1000;
        const dtype dt = 1e-2;
        Matrix<dtype,1,1> dtM;
        dtM(0,0) = dt;
        Matrix<dtype, 1, -1> tVec(1, bufferLength);
        Matrix<dtype, dimTot, -1> xTraj(dimTot, bufferLength);

        Matrix<dtype, dimTot, 1> xCdem, xCctrl;
        myDiffeoControl.getPosDemTot(xCdem);
        myDiffeoControl.getPosDemTot(xCctrl);

        xTraj.col(0) = xCdem;

        double oldNorm = 1e200;
        double newNorm = 1e199;
        Matrix<dtype, dimTot, 1> xDemTest, xCtrlTest, vDemTest, vCtrlTest, dirCtrlTest;

        while (xCctrl.squaredNorm()>(1.e-2*1.e-2)) {

            if (counter%50 == 0){
                cout << "current ctrl pos is " << endl << xCctrl << endl << "for t" << ((double)counter)*dt << endl;
            }

            counter++;

            //Check buffer size
            if (counter >= xTraj.cols()) {
                tVec.conservativeResize(NoChange, xTraj.cols() + bufferLength);
                xTraj.conservativeResize(NoChange, xTraj.cols() + bufferLength);
            }
            if (outInDem) {
                myDiffeoControl.setPosDemTot(xCdem);
            } else {
                myDiffeoControl.setPosCtrlTot(xCdem);
            }
            xCdem = myDiffeoControl.getTraj(dtM, outInDem);

            tVec(0,counter) = counter*dt;
            xTraj.col(counter) = xCdem;

            //Update if converged
            myDiffeoControl.getPosCtrlTot(xCctrl);
            //test
            newNorm = xCctrl.squaredNorm();
            if ((oldNorm-newNorm)<1e-6){
                xDemTest = xCdem;
                xCtrlTest = xCctrl;
                myDiffeoControl.getDemSpaceVel(xDemTest, vDemTest);
                funcDir(xCctrl, vCtrlTest);

                dirCtrlTest = vCtrlTest/(vCtrlTest.norm()+dtype_eps);
                cout << "not converging at ctrl " << endl << xCctrl << endl << "dem " << endl << xCdem << endl << "with dem vel " << endl << vDemTest << endl << " and dir " << endl << vCtrlTest << endl << "jac" << endl
                     << myDiffeoControl._cStateJac << endl;
                cout << "dir ctrl " << endl << dirCtrlTest << endl;
                myConv(xCtrlTest, dirCtrlTest);
                cout << "dir ctrl 2 " << endl << dirCtrlTest << endl;

                cout << "pos dem 1" << xCdem << endl;
                cout << "pos dem 2" << myDiffeoControl._cStateDem << endl;
                break;
            }
            oldNorm = newNorm;

        }

        //Cut unused
        tVec.conservativeResize(NoChange, counter+1);
        xTraj.conservativeResize(NoChange, counter+1);
        //Save
        WriteMatrix(outputFolder+"traj.txt", xTraj);
        WriteMatrix(outputFolder+"tVec.txt", tVec);
    }
};

template<long dim>
void ctrlSpaceTrajDimed(string definitionFolder, string inputFolder, string outputFolder){

    //typedef combinedLocallyWeightedDirections<dim, locallyWeightedDirections<dim, convergingDirections<dim,1>, gaussianKernel<dim - 1, 1>>, locallyWeightedDirections<dim,
    //        convergingDirections<dim,0>, gaussianKernel<dim - 1, 1>>, minimallyConvergingDirection<dim>>
    //        thisCombinedDir;
    typedef combinedDirDimed<dim> thisCombinedDir;

    checkEndOfPath(definitionFolder);
    checkEndOfPath(inputFolder);
    checkEndOfPath(outputFolder);

    thisCombinedDir myDirModel = getDirModel<dim>(definitionFolder);
    minimallyConvergingDirection<dim> myConv(definitionFolder+"minConv.txt"); //Has to be set; If not necessary than simple set convergence to very high positive value
    //Set conv-functor
    myDirModel._conv = myConv;

    myDirModel.setDynamicsFunction();

    //Out
    Matrix<dtype,dim,-1> xOut;
    Matrix<dtype,1,-1> tOut;

    //Get the input
    Matrix<dtype, dim, 1> xInit = ReadMatrix(inputFolder+"pos.txt");

    //Compute
    myDirModel.getDirTraj(xInit, tOut, xOut);

    //Save
    WriteMatrix(outputFolder+"traj.txt", xOut);
    WriteMatrix(outputFolder+"tVec.txt", tOut);
};

int main(int argc, char* argv[]) {

    //argv
    //0: program
    //1: what to do
    //2: Definitionfolder
    //3: Inputfolder
    //4: Outputfolder
    //>4: Optional

    long taskInt = stol(string(argv[1]));

    switch(taskInt){
        //Forwarded diffeo functions
        case 120: forwardDimed<2,0>(string(argv[2]), string(argv[3]), string(argv[4]));break;
        case 121: forwardDimed<2,1>(string(argv[2]), string(argv[3]), string(argv[4]));break;
        case 130: forwardDimed<3,0>(string(argv[2]), string(argv[3]), string(argv[4]));break;
        case 131: forwardDimed<3,1>(string(argv[2]), string(argv[3]), string(argv[4]));break;
        case 220: forwardVDimed<2,0>(string(argv[2]), string(argv[3]), string(argv[4]));break;
        case 221: forwardVDimed<2,1>(string(argv[2]), string(argv[3]), string(argv[4]));break;
        case 230: forwardVDimed<3,0>(string(argv[2]), string(argv[3]), string(argv[4]));break;
        case 231: forwardVDimed<3,1>(string(argv[2]), string(argv[3]), string(argv[4]));break;
        case 320: inverseDimed<2,0>(string(argv[2]), string(argv[3]), string(argv[4]));break;
        case 321: inverseDimed<2,1>(string(argv[2]), string(argv[3]), string(argv[4]));break;
        case 330: inverseDimed<3,0>(string(argv[2]), string(argv[3]), string(argv[4]));break;
        case 331: inverseDimed<3,1>(string(argv[2]), string(argv[3]), string(argv[4]));break;
        case 420: inverseVDimed<2,0>(string(argv[2]), string(argv[3]), string(argv[4]));break;
        case 421: inverseVDimed<2,1>(string(argv[2]), string(argv[3]), string(argv[4]));break;
        case 430: inverseVDimed<3,0>(string(argv[2]), string(argv[3]), string(argv[4]));break;
        case 431: inverseVDimed<3,1>(string(argv[2]), string(argv[3]), string(argv[4]));break;
            //Get velocities
            //argv[5] is either "dem" or "ctrl" to indicate that input points are in dem/ctrl space
        case 520: demSpaceVelDimed<2,0>(string(argv[2]), string(argv[3]), string(argv[4]), string(argv[5])); break;
        case 521: demSpaceVelDimed<2,1>(string(argv[2]), string(argv[3]), string(argv[4]), string(argv[5])); break;
        case 530: demSpaceVelDimed<3,0>(string(argv[2]), string(argv[3]), string(argv[4]), string(argv[5])); break;
        case 531: demSpaceVelDimed<3,1>(string(argv[2]), string(argv[3]), string(argv[4]), string(argv[5])); break;
        case 620: ctrlSpaceVelDimed<2,0>(string(argv[2]), string(argv[3]), string(argv[4]), string(argv[5])); break;
        case 621: ctrlSpaceVelDimed<2,1>(string(argv[2]), string(argv[3]), string(argv[4]), string(argv[5])); break;
        case 630: ctrlSpaceVelDimed<3,0>(string(argv[2]), string(argv[3]), string(argv[4]), string(argv[5])); break;
        case 631: ctrlSpaceVelDimed<3,1>(string(argv[2]), string(argv[3]), string(argv[4]), string(argv[5])); break;
            //Get trajectories
            //argv[5] is either "dem" or "ctrl" to indicate that input points are in dem/ctrl space
        case 720: demSpaceTrajDimed<2,0>(string(argv[2]), string(argv[3]), string(argv[4]), string(argv[5]), true); break;
        case 721: demSpaceTrajDimed<2,1>(string(argv[2]), string(argv[3]), string(argv[4]), string(argv[5]), true); break;
        case 730: demSpaceTrajDimed<3,0>(string(argv[2]), string(argv[3]), string(argv[4]), string(argv[5]), true); break;
        case 731: demSpaceTrajDimed<3,1>(string(argv[2]), string(argv[3]), string(argv[4]), string(argv[5]), true); break;
        case 820: demSpaceTrajDimed<2,0>(string(argv[2]), string(argv[3]), string(argv[4]), string(argv[5]), false); break;
        case 821: demSpaceTrajDimed<2,1>(string(argv[2]), string(argv[3]), string(argv[4]), string(argv[5]), false); break;
        case 830: demSpaceTrajDimed<3,0>(string(argv[2]), string(argv[3]), string(argv[4]), string(argv[5]), false); break;
        case 831: demSpaceTrajDimed<3,1>(string(argv[2]), string(argv[3]), string(argv[4]), string(argv[5]), false); break;
            //Get controlspace trajectories only based on directions
        case 920: ctrlSpaceTrajDimed<2>(string(argv[2]), string(argv[3]), string(argv[4])); break;
        case 930: ctrlSpaceTrajDimed<3>(string(argv[2]), string(argv[3]), string(argv[4])); break;
        case 921: ctrlSpaceTrajDimed<2>(string(argv[2]), string(argv[3]), string(argv[4])); break;
        case 931: ctrlSpaceTrajDimed<3>(string(argv[2]), string(argv[3]), string(argv[4])); break;


        default: assert(false && "Could not be matched");
    }
}