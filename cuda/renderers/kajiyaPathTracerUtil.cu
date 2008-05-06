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
__global__ void k0(const CudaDifferentialGeometry *dg,
                   const bool *stencil,
                   float4 *originsAndMinT)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(stencil[i])
  {
    const CudaDifferentialGeometry &diffG = dg[i];
    const float3 &x = diffG.getPoint();

    originsAndMinT[i] = make_float4(x.x, x.y, x.z, 0.005f);
  } // end if
} // end toRayOrigins()

void differentialGeometryToRayOrigins(const device_ptr<const CudaDifferentialGeometry> &dg,
                                      const device_ptr<const bool> &stencil,
                                      const device_ptr<float4> &originsAndMinT,
                                      const size_t n)
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / BLOCK_SIZE;

  if(gridSize)
    k0<<<gridSize,BLOCK_SIZE>>>(dg, stencil, originsAndMinT);
  if(n%BLOCK_SIZE)
    k0<<<1,n%BLOCK_SIZE>>>(dg + gridSize*BLOCK_SIZE,
                           stencil + gridSize*BLOCK_SIZE,
                           originsAndMinT + gridSize*BLOCK_SIZE);
} // end differentialGeometryToRayOrigins()

// XXX cuda 64b bug forces this short name
__global__ void k1(const CudaDifferentialGeometry *dg,
                   const float minT,
                   const float maxT,
                   const bool *stencil,
                   float4 *originsAndMinT,
                   float4 *directionsAndMaxT)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(stencil[i])
  {
    const CudaDifferentialGeometry &diffG = dg[i];
    const float3 &x = diffG.getPoint();

    originsAndMinT[i] = make_float4(x.x, x.y, x.z, minT);
    directionsAndMaxT[i].w = maxT;
  } // end if
} // end finalizeKernel()

void finalizeRays(const device_ptr<const CudaDifferentialGeometry> &dg,
                  const float minT,
                  const float maxT,
                  const device_ptr<const bool> &stencil,
                  const device_ptr<float4> &originsAndMinT,
                  const device_ptr<float4> &directionsAndMaxT,
                  const size_t n)
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / BLOCK_SIZE;

  if(gridSize)
    k1<<<gridSize,BLOCK_SIZE>>>(dg, minT, maxT, stencil, originsAndMinT, directionsAndMaxT);
  if(n%BLOCK_SIZE)
    k1<<<1,n%BLOCK_SIZE>>>(dg + gridSize*BLOCK_SIZE,
                                       minT, maxT,
                                       stencil + gridSize*BLOCK_SIZE,
                                       originsAndMinT + gridSize*BLOCK_SIZE,
                                       directionsAndMaxT + gridSize*BLOCK_SIZE);
} // end finalizeRaysKernel()

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

struct Points
{
  const CudaIntersection *intersections;
  const CudaDifferentialGeometry *dg;
}; // end Points

//__global__ void k3(const CudaIntersection *intersections,
//                   const CudaDifferentialGeometry *dg,
__global__ void k3(const Points p,
                   const float epsilon,
                   const bool *stencil,
                   float4 *originsAndMinT,
                   float4 *directionsAndMaxT)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(stencil[i])
  {
    // we start at the intersection point
    float3 origin = p.intersections[i].getDifferentialGeometry().getPoint();

    // aim at the point on the light
    float3 dir = p.dg[i].getPoint() - origin;

    originsAndMinT[i]    = make_float4(origin.x, origin.y, origin.z, epsilon);
    directionsAndMaxT[i] = make_float4(dir.x, dir.y, dir.z, 1.0f - epsilon);
  } // end if
} // end k3()

void createShadowRays(const device_ptr<const CudaIntersection> &intersections,
                      const device_ptr<const CudaDifferentialGeometry> &dg,
                      const float epsilon,
                      const device_ptr<const bool> &stencil,
                      const device_ptr<float4> &originsAndMinT,
                      const device_ptr<float4> &directionsAndMaxT,
                      const size_t n)
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / BLOCK_SIZE;

  Points p = {intersections, dg};

  if(gridSize)
    k3<<<gridSize,BLOCK_SIZE>>>(p,epsilon,stencil,originsAndMinT,directionsAndMaxT);
  if(n%BLOCK_SIZE)
  {
    p.intersections += gridSize*BLOCK_SIZE;
    p.dg += gridSize*BLOCK_SIZE;

    k3<<<1,n%BLOCK_SIZE>>>(p,
                           epsilon,
                           stencil + gridSize*BLOCK_SIZE,
                           originsAndMinT + gridSize*BLOCK_SIZE,
                           directionsAndMaxT + gridSize*BLOCK_SIZE);
  } // end if
} // end createShadowRays()

__global__ void k4(const float4 *directionsAndMaxT,
                   const bool *stencil,
                   float3 *wo)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(stencil[i])
  {
    float4 rayDir = directionsAndMaxT[i];
    wo[i] = make_float3(-rayDir.x, -rayDir.y, -rayDir.z);
  } // end if
} // end k4()

void flipRayDirections(const stdcuda::device_ptr<const float4> &directionsAndMaxT,
                       const stdcuda::device_ptr<const bool> &stencil,
                       const stdcuda::device_ptr<float3> &wo,
                       const size_t n)
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / BLOCK_SIZE;

  if(gridSize)
    k4<<<gridSize,BLOCK_SIZE>>>(directionsAndMaxT, stencil, wo);
  if(n%BLOCK_SIZE)
    k4<<<1,n%BLOCK_SIZE>>>(directionsAndMaxT + gridSize*BLOCK_SIZE,
                           stencil + gridSize*BLOCK_SIZE,
                           wo + gridSize*BLOCK_SIZE);
} // end flipRayDirections()

__global__ void k5(const float3 *wo,
                   const CudaDifferentialGeometry *dg,
                   const bool *stencil,
                   float *pdf)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(stencil[i])
  {
    float3 n = dg[i].getNormal();

    // divide by the absolute value of the cosine
    // between n and wo
    pdf[i] /= fabs(dot(n,wo[i]));
  } // end if
} // end k5()

void toProjectedSolidAnglePdf(const device_ptr<const float3> &wo,
                              const device_ptr<const CudaDifferentialGeometry> &dg,
                              const device_ptr<const bool> &stencil,
                              const device_ptr<float> &pdf,
                              const size_t n)
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / BLOCK_SIZE;

  if(gridSize)
    k5<<<gridSize,BLOCK_SIZE>>>(wo, dg, stencil, pdf);
  if(n%BLOCK_SIZE)
    k5<<<1,n%BLOCK_SIZE>>>(wo + gridSize*BLOCK_SIZE,
                           dg + gridSize*BLOCK_SIZE,
                           stencil + gridSize*BLOCK_SIZE,
                           pdf + gridSize*BLOCK_SIZE);
} // end toProjectedSolidAnglePdf()

__global__ void k6(const float4 *directionsAndMaxT,
                   const bool *stencil,
                   float3 *w)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(stencil[i])
  {
    float4 dir = directionsAndMaxT[i];
    w[i] = make_float3(dir.x, dir.y, dir.z);
  } // end if
} // end k6()

void rayDirectionsToFloat3(const device_ptr<const float4> &directionsAndMaxT,
                           const device_ptr<const bool> &stencil,
                           const device_ptr<float3> &w,
                           const size_t n)
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / BLOCK_SIZE;

  if(gridSize)
    k6<<<gridSize,BLOCK_SIZE>>>(directionsAndMaxT, stencil, w);
  if(n%BLOCK_SIZE)
    k6<<<1,n%BLOCK_SIZE>>>(directionsAndMaxT + gridSize*BLOCK_SIZE,
                           stencil + gridSize*BLOCK_SIZE,
                           w + gridSize*BLOCK_SIZE);
} // end rayDirectionsToFloat3()

__global__ void k7(const CudaDifferentialGeometry *dg0,
                   const int dg0Stride,
                   const CudaDifferentialGeometry *dg1,
                   const bool *stencil,
                   float *g)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(stencil[i])
  {
    const char *ptr = reinterpret_cast<const char*>(dg0) + i * dg0Stride;
    const CudaDifferentialGeometry &diffG0 = *reinterpret_cast<const CudaDifferentialGeometry*>(ptr);

    float3 p0 = diffG0.getPoint();
    float3 p1 = dg1[i].getPoint();

    // compute the vector pointing from p0 to p1
    float3 w = p1 - p0;

    // compute squared distance
    float d2 = dot(w,w);

    // normalize w
    w /= sqrtf(d2);

    float3 n0 = diffG0.getNormal();
    float3 n1 = dg1[i].getNormal();

    g[i] = fabs(dot(n0,w)) * fabs(dot(n1,w)) / d2;
  } // end if
} // end k7()

void evaluateGeometricTerm(const device_ptr<const CudaDifferentialGeometry> &dg0,
                           const size_t dg0Stride,
                           const device_ptr<const CudaDifferentialGeometry> &dg1,
                           const device_ptr<const bool> &stencil,
                           const device_ptr<float> &g,
                           const size_t n)
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / BLOCK_SIZE;

  if(gridSize)
    k7<<<gridSize,BLOCK_SIZE>>>(dg0,dg0Stride,dg1,stencil,g);

  if(n%BLOCK_SIZE)
  {
    k7<<<1,n%BLOCK_SIZE>>>(stride_cast(dg0.get(), gridSize*BLOCK_SIZE, dg0Stride),
                           dg0Stride,
                           dg1 + gridSize*BLOCK_SIZE,
                           stencil + gridSize*BLOCK_SIZE,
                           g + gridSize*BLOCK_SIZE);
  } // end if
} // end evaluateGeometricTerm()

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

//__global__ void k14(const CudaIntersection *intersections,
//                    CudaDifferentialGeometry *dg)
//{
//  int i = blockIdx.x * blockDim.x + threadIdx.x;
//  dg[i] = intersections[i].getDifferentialGeometry();
//} // end k14()
//
void copyDifferentialGeometry(const device_ptr<const CudaIntersection> &intersections,
                              const device_ptr<CudaDifferentialGeometry> &dg,
                              const size_t n)
{
  //unsigned int BLOCK_SIZE = 192;
  //unsigned int gridSize = n / BLOCK_SIZE;

  //if(gridSize)
  //  k14<<<gridSize,BLOCK_SIZE>>>(intersections,dg);
  //if(n%BLOCK_SIZE)
  //  k14<<<1,n%BLOCK_SIZE>>>(intersections + gridSize*BLOCK_SIZE,
  //                          dg + gridSize*BLOCK_SIZE);

  const void *temp = &intersections[0];
  device_ptr<const CudaDifferentialGeometry> firstDg(reinterpret_cast<const CudaDifferentialGeometry*>(temp));
  stdcuda::stride_copy(firstDg, sizeof(CudaIntersection), dg, sizeof(CudaDifferentialGeometry), n);
} // end copyDifferentialGeometry()

