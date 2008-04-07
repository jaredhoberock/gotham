/*! \file evaluateBidirectionalScattering.cu
 *  \author Jared Hoberock
 *  \brief Implementation of evaluateBidirectionalScattering kernel.
 */

// XXX hack total
#define inline inline __host__ __device__

#include <stdcuda/vector_math.h>
#include "CudaScatteringDistributionFunction.h"
#include "evaluateBidirectionalScattering.h"

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

void __global__ ks(const CudaScatteringDistributionFunction *f,
                   const float3 *wo,
                   const CudaDifferentialGeometry *dg,
                   const int dgStride,
                   const float3 *wi,
                   const int *stencil,
                   float3 *results)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(stencil[i])
  {
    const char *ptr = reinterpret_cast<const char*>(dg) + i*dgStride;
    const CudaDifferentialGeometry *diffG = reinterpret_cast<const CudaDifferentialGeometry*>(ptr);
    results[i] = f[i].evaluate(wo[i], *diffG, wi[i]);
  } // end if
} // end kernel()

void evaluateBidirectionalScattering(const CudaScatteringDistributionFunction *f,
                                     const float3 *wo,
                                     const CudaDifferentialGeometry *dg,
                                     const float3 *wi,
                                     const int *stencil,
                                     float3 *results,
                                     const size_t n)
{
  dim3 grid = dim3(1,1,1);
  dim3 block = dim3(n,1,1);

  kernel<<<grid,block>>>(f,wo,dg,wi,stencil,results);
} // end evaluateBidirectionalScattering()

void evaluateBidirectionalScatteringStride(const CudaScatteringDistributionFunction *f,
                                           const float3 *wo,
                                           const CudaDifferentialGeometry *dg,
                                           const size_t dgStride,
                                           const float3 *wi,
                                           const int *stencil,
                                           float3 *results,
                                           const size_t n)
{
  dim3 grid = dim3(1,1,1);
  dim3 block = dim3(n,1,1);

  ks<<<grid,block>>>(f,wo,dg,dgStride,wi,stencil,results);
} // end evaluateBidirectionalScattering()

