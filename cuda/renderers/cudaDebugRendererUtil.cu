/*! \file cudaDebugRendererUtil.cu
 *  \author Jared Hoberock
 *  \brief Implementation of cudaDebugRendererUtil functions.
 */

#include "cudaDebugRendererUtil.h"
#include <stdcuda/vector_math.h>

using namespace stdcuda;

__global__ void sum(const float3 *s,
                    const float3 *e,
                    const bool *stencil,
                    float3 *r)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(stencil[i])
  {
    r[i] = s[i] + e[i];
  } // end if
} // end sum()

void sumScatteringAndEmission(const device_ptr<const float3> &scattering,
                              const device_ptr<const float3> &emission,
                              const device_ptr<const bool> &stencil,
                              const device_ptr<float3> &result,
                              const size_t n)
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / BLOCK_SIZE;

  if(gridSize)
    sum<<<gridSize,BLOCK_SIZE>>>(scattering,emission,stencil,result);
  if(n%BLOCK_SIZE)
    sum<<<1,n%BLOCK_SIZE>>>(scattering + gridSize*BLOCK_SIZE,
                            emission + gridSize*BLOCK_SIZE,
                            stencil + gridSize*BLOCK_SIZE,
                            result + gridSize*BLOCK_SIZE);
} // end sumScatteringAndEmission()

__global__ void flipVectorsKernel(const float3 *src,
                                  float3 *dst)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float3 v = src[i];
  dst[i] = make_float3(-v.x, -v.y, -v.z);
} // end flipVectorsKernel()

void flipVectors(const device_ptr<const float3> &src,
                 const device_ptr<float3> &result,
                 const size_t n)
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / BLOCK_SIZE;

  if(gridSize)
    flipVectorsKernel<<<gridSize,BLOCK_SIZE>>>(src,result);
  if(n%BLOCK_SIZE)
    flipVectorsKernel<<<1,n%BLOCK_SIZE>>>(src + gridSize*BLOCK_SIZE,
                                          result + gridSize*BLOCK_SIZE);
} // end flipVectors()

