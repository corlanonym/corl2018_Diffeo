cmake_minimum_required(VERSION 3.0)
project(cpp)

SET(CMAKE_CXX_COMPILER /usr/bin/g++)

#find_package (Eigen3 REQUIRED NO_MODULE)
#include_directories(${EIGEN3_INCLUDE_DIRS})

include_directories( ./includes )
include_directories( ./src )

set(CMAKE_CXX_FLAGS "-msse2 -O3")
#set(CMAKE_CXX_FLAGS "-msse2")

set(CMAKE_CXX_STANDARD 11)

#add_executable(cppTest test.cpp src/gmm.h src/diffeoUtils.h src/diffeoCtrl.h src/fileVector.h src/fileVector.cpp src/convergenceUtils.h src/schlepilUtils.h src/schlepilUtils.cpp)

#add_executable(cpp main.cpp src/gmm.h src/diffeoUtils.h src/diffeoCtrl.h src/fileVector.h src/fileVector.cpp src/convergenceUtils.h src/schlepilUtils.h src/schlepilUtils.cpp)

#add_executable(compTraj computeTraj.cpp src/gmm.h src/diffeoUtils.h src/diffeoCtrl.h src/fileVector.h src/fileVector.cpp src/convergenceUtils.h src/schlepilUtils.h src/schlepilUtils.cpp src/polyEval.h src/combinedDynamics.h src/myUtils.h combinedDynUnitTest.cpp)

#add_executable(utils utils.cpp src/gmm.h src/diffeoUtils.h src/diffeoCtrl.h src/fileVector.h src/fileVector.cpp src/convergenceUtils.h src/schlepilUtils.h src/schlepilUtils.cpp)

set( diffeoSRC ./src/diffeoLib.h ./src/schlepilUtils.h ./src/schlepilUtils.cpp ./src/fileVector.h ./src/fileVector.cpp)

add_executable(diffeoInterface diffeoInterface.cpp ${diffeoSRC})

set(VERBOSE 1)
