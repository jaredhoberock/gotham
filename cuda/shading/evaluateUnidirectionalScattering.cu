/*! \file evaluateUnidirectionalScattering.cu
 *  \author Jared Hoberock
 *  \brief Implementation of evaluateUnidirectionalScattering kernel.
 */

// XXX hack total
#define inline inline __host__ __device__

#include "evaluateUnidirectionalScattering.h"
#include "CudaScatteringDistributionFunction.h"
#include "CudaLambertian.h"
#include "../geometry/CudaDifferentialGeometry.h"
#include <spectrum/Spectrum.h>

#undef inline

// XXX BUG: give this a short name thanks to
//     this bug: 
//     http://forums.nvidia.com/index.php?showtopic=53767
static void __global__ k(const CudaScatteringDistributionFunction *f,
                         const float3 *wo,
                         const CudaDifferentialGeometry *dg,
                         const int *stencil,
                         float3 *results)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(stencil[i])
  {
    results[i] = f[i].evaluate(wo[i], dg[i]);
  } // end if
} // end kernel()

void evaluateUnidirectionalScattering(const CudaScatteringDistributionFunction *f,
                                      const float3 *wo,
                                      const CudaDifferentialGeometry *dg,
                                      const int *stencil,
                                      Spectrum *results,
                                      const size_t n)
{
  dim3 grid = dim3(1,1,1);
  dim3 block = dim3(n,1,1);

  k<<<grid,block>>>(f,wo,dg,stencil,(float3*)results);
} // end evaluateUnidirectionalScattering()

