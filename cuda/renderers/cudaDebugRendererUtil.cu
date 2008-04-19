/*! \file cudaDebugRendererUtil.cu
 *  \author Jared Hoberock
 *  \brief Implementation of cudaDebugRendererUtil functions.
 */

#include "cudaDebugRendererUtil.h"
#include <stdcuda/vector_math.h>

using namespace stdcuda;

__global__ void kernel(const CudaDifferentialGeometry *dg,
                       float4 *originsAndMinT,
                       float4 *directionsAndMaxT)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  const CudaDifferentialGeometry &diffG = dg[i];
  const float3 &x = diffG.getPoint();

  originsAndMinT[i] = make_float4(x.x, x.y, x.z, 0.005f);
  directionsAndMaxT[i].w = 10000000.0f;
} // end kernel()

void sampleEyeRaysEpilogue(const device_ptr<const CudaDifferentialGeometry> &dg,
                           const device_ptr<float4> &originsAndMinT,
                           const device_ptr<float4> &directionsAndMaxT,
                           const size_t n)
{
  dim3 grid(1,1,1);
  dim3 block(n,1,1);

  kernel<<<grid,block>>>(dg,originsAndMinT,directionsAndMaxT);
} // end sampleEyeRaysEpilogue()

__global__ void toWo(const float4 *directionsAndMaxT,
                     float3 *wo)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  float4 dir = directionsAndMaxT[i];

  float3 woOut = make_float3(-dir.x, -dir.y, -dir.z);
  wo[i] = woOut;
} // end toWo()

void rayDirectionsToWo(const device_ptr<const float4> &directionsAndMaxT,
                       const device_ptr<float3> &wo,
                       const size_t n)
{
  dim3 grid(1,1,1);
  dim3 block(n,1,1);

  toWo<<<grid,block>>>(directionsAndMaxT, wo);
} // end rayDirectionsToWo()

__global__ void sum(const float3 *s,
                    const float3 *e,
                    const int *stencil,
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
                              const device_ptr<const int> &stencil,
                              const device_ptr<float3> &result,
                              const size_t n)
{
  dim3 grid(1,1,1);
  dim3 block(n,1,1);

  sum<<<grid,block>>>(scattering,emission,stencil,result);
} // end sumScatteringAndEmission()

