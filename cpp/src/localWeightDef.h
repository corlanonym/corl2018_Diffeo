//
// Created by elfuius on 08/06/18.
//

#ifndef PROJECT_LOCALWEIGHTDEF_H
#define PROJECT_LOCALWEIGHTDEF_H

#define whichKernelForWeight_ 0

//Use gaussian kernels as weight functions
#if whichKernelForWeight_ == 0
    #include "gaussianKernel.h"
    //template typedef -> not possible so alias
    template <long dim>
    using weightKernel=gaussianKernel<dim-1,1>;
    using weight2d = gaussianKernel<1,1>;
    using weight3d = gaussianKernel<2,1>;
#elif whichKernelForWeight_ == 1
    #include "cauchyKernel.h"
    //template typedef -> not possible so alias
    template <long dim>
    using weightKernel=cauchyKernel<dim>;
    using weight2d = cauchyKernel<2>;
    using weight3d = cauchyKernel<3>;
#else
#error Kernel not understood
#endif

#endif //PROJECT_LOCALWEIGHTDEF_H
