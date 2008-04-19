/*! \file evaluateBidirectionalScattering.cu
 *  \author Jared Hoberock *  \brief Implementation of evaluateBidirectionalScattering kernel.
 */

// XXX hack total
#define inline inline __host__ __device__

#include <stdcuda/vector_math.h>
#include "CudaScatteringDistributionFunction.h"
#include "evaluateBidirectionalScattering.h"
#include <stdcuda/stride_cast.h>

#undef inline

using namespace stdcuda;

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
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / BLOCK_SIZE;

  if(gridSize)
    kernel<<<gridSize,BLOCK_SIZE>>>(f,wo,dg,wi,stencil,results);
  if(n%BLOCK_SIZE)
    kernel<<<1,n%BLOCK_SIZE>>>(f + gridSize*BLOCK_SIZE,
                               wo + gridSize*BLOCK_SIZE,
                               dg + gridSize*BLOCK_SIZE,
                               wi + gridSize*BLOCK_SIZE,
                               stencil + gridSize*BLOCK_SIZE,
                               results + gridSize*BLOCK_SIZE);
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
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / BLOCK_SIZE;

  if(gridSize)
    ks<<<gridSize,BLOCK_SIZE>>>(f,wo,dg,dgStride,wi,stencil,results);
  if(n%BLOCK_SIZE)
    ks<<<1,n%BLOCK_SIZE>>>(f + gridSize*BLOCK_SIZE,
                           wo + gridSize*BLOCK_SIZE,
                           stride_cast(dg,gridSize*BLOCK_SIZE,dgStride),
                           dgStride,
                           wi + gridSize*BLOCK_SIZE,
                           stencil + gridSize*BLOCK_SIZE,
                           results + gridSize*BLOCK_SIZE);
} // end evaluateBidirectionalScattering()

