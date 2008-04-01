/*! \file evaluateBidirectionalScattering.cu
 *  \author Jared Hoberock
 *  \brief Implementation of evaluateBidirectionalScattering kernel.
 */

// XXX hack total
#define inline inline __host__ __device__

#include "../geometry/CudaDifferentialGeometry.h"
#include "evaluateBidirectionalScattering.h"
#include "CudaScatteringDistributionFunction.h"
#include "CudaLambertian.h"
#include <spectrum/Spectrum.h>

#undef inline

void __global__ kernel(const CudaScatteringDistributionFunction *f,
                       const float3 *wo,
                       const CudaDifferentialGeometry *dg,
                       const float3 *wi,
                       const int *stencil,
                       float3 *results)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(stencil[i])
  {
    results[i] = f[i].evaluate(wo[i], dg[i], wi[i]);
  } // end if
} // end kernel()

void evaluateBidirectionalScattering(const CudaScatteringDistributionFunction *f,
                                     const float3 *wo,
                                     const CudaDifferentialGeometry *dg,
                                     const float3 *wi,
                                     const int *stencil,
                                     Spectrum *results,
                                     const size_t n)
{
  dim3 grid = dim3(1,1,1);
  dim3 block = dim3(n,1,1);

  kernel<<<grid,block>>>(f,wo,dg,wi,stencil,(float3*)results);
} // end evaluateBidirectionalScattering()

