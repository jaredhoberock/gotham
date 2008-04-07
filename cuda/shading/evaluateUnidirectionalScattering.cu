/*! \file evaluateUnidirectionalScattering.cu
 *  \author Jared Hoberock
 *  \brief Implementation of evaluateUnidirectionalScattering kernel.
 */

// XXX hack total
#define inline inline __host__ __device__

#include <stdcuda/vector_math.h>
#include "CudaScatteringDistributionFunction.h"
#include "evaluateUnidirectionalScattering.h"

#undef inline

void __global__ ks(const CudaScatteringDistributionFunction *f,
                   const float3 *wo,
                   const CudaDifferentialGeometry *dg,
                   const int dgStride,
                   const int *stencil,
                   float3 *results)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(stencil[i])
  {
    const char *ptr = reinterpret_cast<const char*>(dg) + i*dgStride;
    const CudaDifferentialGeometry *diffG = reinterpret_cast<const CudaDifferentialGeometry*>(ptr);
    results[i] = f[i].evaluate(wo[i], *diffG);
  } // end if
} // end kernel()

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
                                      float3 *results,
                                      const size_t n)
{
  dim3 grid = dim3(1,1,1);
  dim3 block = dim3(n,1,1);

  k<<<grid,block>>>(f,wo,dg,stencil,(float3*)results);
} // end evaluateUnidirectionalScattering()

void evaluateUnidirectionalScatteringStride(const CudaScatteringDistributionFunction *f,
                                            const float3 *wo,
                                            const CudaDifferentialGeometry *dg,
                                            const size_t dgStride,
                                            const int *stencil,
                                            float3 *results,
                                            const size_t n)
{
  dim3 grid = dim3(1,1,1);
  dim3 block = dim3(n,1,1);

  ks<<<grid,block>>>(f,wo,dg,dgStride,stencil,results);
} // end evaluateUnidirectionalScattering()

