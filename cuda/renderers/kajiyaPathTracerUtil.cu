/*! \file kajiyaPathTracerUtil.cu
 *  \author Jared Hoberock
 *  \brief Implementation of the functions declared
 *         in kajiyaPathTracerUtil.h.
 */

#include "kajiyaPathTracerUtil.h"
#include <stdcuda/vector_math.h>
#include <stdcuda/stride_cast.h>
#include <stdcuda/stride_copy.h>

using namespace stdcuda;

// XXX cuda 64b bug forces this short name
__global__ void k2(const float3 *scattering,
                   const float *pdfs,
                   const bool *stencil,
                   float3 *throughput)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(stencil[i])
  {
    throughput[i] *= scattering[i] / pdfs[i];
  } // end if
} // end k2()

void updateThroughput(const device_ptr<const float3> &scattering,
                      const device_ptr<const float> &pdfs,
                      const device_ptr<const bool> &stencil,
                      const device_ptr<float3> &throughput,
                      const size_t n)
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / BLOCK_SIZE;

  if(gridSize)
    k2<<<gridSize,BLOCK_SIZE>>>(scattering, pdfs, stencil, throughput);
  if(n%BLOCK_SIZE)
    k2<<<1,n%BLOCK_SIZE>>>(scattering + gridSize*BLOCK_SIZE,
                           pdfs + gridSize*BLOCK_SIZE,
                           stencil + gridSize*BLOCK_SIZE,
                           throughput + gridSize*BLOCK_SIZE);
} // end updateThroughput()

__global__ void k8(const float weight,
                   const float3 *throughput,
                   const float3 *scattering,
                   const float3 *emission,
                   const float *geometricTerm,
                   const float *pdf,
                   const bool *stencil,
                   float3 *result)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(stencil[i])
  {
    result[i] += weight * throughput[i] * scattering[i] * emission[i] * geometricTerm[i] / pdf[i];
  } // end if
} // end k8()

void accumulateLightSample(const float weight,
                           const device_ptr<const float3> &throughput,
                           const device_ptr<const float3> &scattering,
                           const device_ptr<const float3> &emission,
                           const device_ptr<const float> &geometricTerm,
                           const device_ptr<const float> &pdf,
                           const device_ptr<const bool> &stencil,
                           const device_ptr<float3> &result,
                           const size_t n)
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / BLOCK_SIZE;

  if(gridSize)
    k8<<<gridSize,BLOCK_SIZE>>>(weight,throughput,scattering,emission,geometricTerm,pdf,stencil,result);
  if(n%BLOCK_SIZE)
    k8<<<1,n%BLOCK_SIZE>>>(weight,
                           throughput + gridSize*BLOCK_SIZE,
                           scattering + gridSize*BLOCK_SIZE,
                           emission + gridSize*BLOCK_SIZE,
                           geometricTerm + gridSize*BLOCK_SIZE,
                           pdf + gridSize*BLOCK_SIZE,
                           stencil + gridSize*BLOCK_SIZE,
                           result + gridSize*BLOCK_SIZE);
} // end accumulateLightSample()

__global__ void k9(const float4 *x,
                   float2 *y)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float4 temp = x[i];
  y[i] = make_float2(temp.x, temp.y);
} // end k9()

void extractFloat2(const stdcuda::device_ptr<const float4> &x,
                   const stdcuda::device_ptr<float2> &y,
                   const size_t n)
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / BLOCK_SIZE;

  if(gridSize)
    k9<<<gridSize,BLOCK_SIZE>>>(x,y);
  if(n%BLOCK_SIZE)
    k9<<<1,n%BLOCK_SIZE>>>(x + gridSize*BLOCK_SIZE,
                           y + gridSize*BLOCK_SIZE);
} // end extractFloat2()

__global__ void k10(const float3 *throughput,
                    const float3 *emission,
                    const bool *stencil,
                    float3 *result)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(stencil[i])
  {
    result[i] += throughput[i] * emission[i];
  } // end if
} // end accumulateEmission()

void accumulateEmission(const device_ptr<const float3> &throughput,
                        const device_ptr<const float3> &emission,
                        const device_ptr<const bool> &stencil,
                        const device_ptr<float3> &result,
                        const size_t n )
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / BLOCK_SIZE;

  if(gridSize)
    k10<<<gridSize,BLOCK_SIZE>>>(throughput,emission,stencil,result);
  if(n%BLOCK_SIZE)
    k10<<<1,n%BLOCK_SIZE>>>(throughput + gridSize*BLOCK_SIZE,
                            emission + gridSize*BLOCK_SIZE,
                            stencil + gridSize*BLOCK_SIZE,
                            result + gridSize*BLOCK_SIZE);
} // end accumulateEmission()

__global__ void k11(const float *pdf,
                    float3 *throughput)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  throughput[i] /= pdf[i];
} // end k11()

void divideByPdf(const device_ptr<const float> &pdf,
                 const device_ptr<float3> &throughput,
                 const size_t n)
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / BLOCK_SIZE;

  if(gridSize)
    k11<<<gridSize,BLOCK_SIZE>>>(pdf,throughput);
  if(n%BLOCK_SIZE)
    k11<<<1,n%BLOCK_SIZE>>>(pdf + gridSize*BLOCK_SIZE,
                            throughput + gridSize*BLOCK_SIZE);
} // end divideByPdf()

__global__ void k12(float3 *w,
                    const bool *stencil)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(stencil[i])
  {
    float3 temp = w[i];
    temp.x = -temp.x;
    temp.y = -temp.y;
    temp.z = -temp.z;
    w[i] = temp;
  } // end if
} // end k12()

void flipVectors(const device_ptr<float3> &w,
                 const device_ptr<const bool> &stencil,
                 const size_t n)
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / BLOCK_SIZE;

  if(gridSize)
    k12<<<gridSize,BLOCK_SIZE>>>(w,stencil);
  if(n%BLOCK_SIZE)
    k12<<<1,n%BLOCK_SIZE>>>(w + gridSize*BLOCK_SIZE,
                            stencil + gridSize*BLOCK_SIZE);
} // end flipVectors()

__global__ void k13(const unsigned int w, const unsigned int h,
                    float4 *u,
                    const float2 invPixelSize,
                    const unsigned int offset)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x + offset;

  float4 temp = u[i];

  temp.x = static_cast<float>(i % w) / w + temp.x * invPixelSize.x;
  temp.y = static_cast<float>(i / w) / h + temp.y * invPixelSize.y;

  u[i] = temp;
} // end k13()

void stratify(const unsigned int w, const unsigned int h,
              const device_ptr<float4> &u,
              const size_t n)
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / BLOCK_SIZE;

  float2 invPixelSize = make_float2(1.0f / w, 1.0f / h);

  if(gridSize)
    k13<<<gridSize,BLOCK_SIZE>>>(w, h, u, invPixelSize, 0);
  if(n%BLOCK_SIZE)
    k13<<<1,n%BLOCK_SIZE>>>(w, h, u, invPixelSize, gridSize*BLOCK_SIZE);
} // end stratify()

__global__ void k14(const float3 *src,
                    const bool *stencil,
                    float3 *dst)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  float3 v = src[i];
  dst[i] = make_float3(-v.x, -v.y, -v.z);
} // end k14()

void flipVectors(const device_ptr<const float3> &src,
                 const device_ptr<const bool> &stencil,
                 const device_ptr<float3> &dst,
                 const size_t n)
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / BLOCK_SIZE;

  if(gridSize)
    k14<<<gridSize,BLOCK_SIZE>>>(src, stencil, dst);
  if(n%BLOCK_SIZE)
    k14<<<1,n%BLOCK_SIZE>>>(src + gridSize*BLOCK_SIZE,
                            stencil + gridSize*BLOCK_SIZE,
                            dst + gridSize*BLOCK_SIZE);
} // end flipVectors()

__global__ void k15(const float3 *from,
                    const float3 *to,
                    const float epsilon,
                    const bool *stencil,
                    float3 *directions)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(stencil[i])
  {
    // we start at the intersection point
    float3 origin = from[i];

    // aim at the point on the light
    float3 dir = to[i];
    dir -= origin;

    directions[i] = dir;
  } // end if
} // end k15()

void createShadowRays(const device_ptr<const float3> &from,
                      const device_ptr<const float3> &to,
                      const float epsilon,
                      const device_ptr<const bool> &stencil,
                      const device_ptr<float3> &directions,
                      const size_t n)
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / BLOCK_SIZE;

  if(gridSize)
    k15<<<gridSize,BLOCK_SIZE>>>(from,
                                 to,
                                 epsilon,
                                 stencil,
                                 directions);
  if(n%BLOCK_SIZE)
  {
    k15<<<1,n%BLOCK_SIZE>>>(from + gridSize*BLOCK_SIZE,
                            to + gridSize*BLOCK_SIZE,
                            epsilon,
                            stencil + gridSize*BLOCK_SIZE,
                            directions + gridSize*BLOCK_SIZE);
  } // end if
} // end createShadowRays()

__global__ void k16(const float3 *x0,
                    const float3 *n0,
                    const float3 *x1,
                    const float3 *n1,
                    const bool *stencil,
                    float *g)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(stencil[i])
  {
    float3 p0 = x0[i];
    float3 p1 = x1[i];

    // compute the vector pointing from p0 to p1
    float3 w = p1 - p0;

    // compute squared distance
    float d2 = dot(w,w);

    // normalize w
    w /= sqrtf(d2);

    float3 normal0 = n0[i];
    float3 normal1 = n1[i];

    g[i] = fabs(dot(normal0,w)) * fabs(dot(normal1,w)) / d2;
  } // end if
} // end k16()

void evaluateGeometricTerm(const device_ptr<const float3> &x0,
                           const device_ptr<const float3> &n0,
                           const device_ptr<const float3> &x1,
                           const device_ptr<const float3> &n1,
                           const device_ptr<const bool> &stencil,
                           const device_ptr<float> &g,
                           const size_t n)
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / BLOCK_SIZE;

  if(gridSize)
    k16<<<gridSize,BLOCK_SIZE>>>(x0,n0,x1,n1,stencil,g);
  if(n%BLOCK_SIZE)
    k16<<<1,n%BLOCK_SIZE>>>(x0 + gridSize*BLOCK_SIZE,
                            n0 + gridSize*BLOCK_SIZE,
                            x1 + gridSize*BLOCK_SIZE,
                            n1 + gridSize*BLOCK_SIZE,
                            stencil + gridSize*BLOCK_SIZE,
                            g + gridSize*BLOCK_SIZE);
} // end evaluateGeometricTerm()

__global__ void k17(const float3 *wo,
                    const float3 *normals,
                    const bool *stencil,
                    float *pdf)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(stencil[i])
  {
    float3 n = normals[i];

    // divide by the absolute value of the cosine
    // between n and wo
    pdf[i] /= fabs(dot(n,wo[i]));
  } // end if
} // end k17()

void toProjectedSolidAnglePdf(const device_ptr<const float3> &wo,
                              const device_ptr<const float3> &normals,
                              const device_ptr<const bool> &stencil,
                              const device_ptr<float> &pdf,
                              const size_t n)
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / BLOCK_SIZE;

  if(gridSize)
    k17<<<gridSize,BLOCK_SIZE>>>(wo,normals,stencil,pdf);
  if(n%BLOCK_SIZE)
    k17<<<1,n%BLOCK_SIZE>>>(wo + gridSize*BLOCK_SIZE,
                            normals + gridSize*BLOCK_SIZE,
                            stencil + gridSize*BLOCK_SIZE,
                            pdf + gridSize*BLOCK_SIZE);
} // end toProjectedSolidAnglePdf()

