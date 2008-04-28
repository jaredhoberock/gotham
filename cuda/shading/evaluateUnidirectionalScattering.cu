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
                   const bool *stencil,
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
                         const bool *stencil,
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
                                      const bool *stencil,
                                      float3 *results,
                                      const size_t n)
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / BLOCK_SIZE;

  if(gridSize)
    k<<<gridSize,BLOCK_SIZE>>>(f,wo,dg,stencil,results);
  if(n%BLOCK_SIZE)
    k<<<1,n%BLOCK_SIZE>>>(f + gridSize*BLOCK_SIZE,
                          wo + gridSize*BLOCK_SIZE,
                          dg + gridSize*BLOCK_SIZE,
                          stencil + gridSize*BLOCK_SIZE,
                          results + gridSize*BLOCK_SIZE);
} // end evaluateUnidirectionalScattering()

void evaluateUnidirectionalScatteringStride(const CudaScatteringDistributionFunction *f,
                                            const float3 *wo,
                                            const CudaDifferentialGeometry *dg,
                                            const size_t dgStride,
                                            const bool *stencil,
                                            float3 *results,
                                            const size_t n)
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / BLOCK_SIZE;

  if(gridSize)
    ks<<<gridSize,BLOCK_SIZE>>>(f,wo,dg,dgStride,stencil,results);
  if(n%BLOCK_SIZE)
  {
    const char *temp = reinterpret_cast<const char*>(dg) + dgStride*gridSize*BLOCK_SIZE;
    ks<<<1,n%BLOCK_SIZE>>>(f + gridSize*BLOCK_SIZE,
                           wo + gridSize*BLOCK_SIZE,
                           reinterpret_cast<const CudaDifferentialGeometry*>(temp),
                           dgStride,
                           stencil + gridSize*BLOCK_SIZE,
                           results + gridSize*BLOCK_SIZE);
  } // end if
} // end evaluateUnidirectionalScattering()

