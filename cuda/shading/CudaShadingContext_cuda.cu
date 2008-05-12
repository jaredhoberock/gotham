/*! \file CudaShadingContext_cuda.cu
 *  \author Jared Hoberock
 *  \brief CUDA implementation of CudaShadingContext class.
 */

// XXX remove this when exceptions are enabled in nvcc
#define BOOST_NO_EXCEPTIONS

#include <stdcuda/stride_cast.h>
#include "CudaShadingContext.h"

#include "CudaScatteringDistributionFunction.h"

using namespace stdcuda;

struct AltUniParameters
{
  const CudaScatteringDistributionFunction *f;
  CudaDifferentialGeometryArray dg;
  const float3 *u;
  int uStride;
  float3 *s;
  float3 *wo;
  int woStride;
  float *pdf;
  bool *delta;
}; // end Parameters

void __global__ sampleUnidirectionalKernel(const AltUniParameters p)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  const char *ptr = reinterpret_cast<const char*>(p.u) + i*p.uStride;
  float3 u = *reinterpret_cast<const float3*>(ptr);

  char *temp = reinterpret_cast<char*>(p.wo) + i*p.woStride;
  float3 &wo = *reinterpret_cast<float3*>(temp);

  // sample
  p.f[i].sample(p.dg.mPoints[i],
                p.dg.mTangents[i],
                p.dg.mBinormals[i],
                p.dg.mNormals[i],
                u.x, u.y, u.z, p.s[i], wo, p.pdf[i], p.delta[i]);
} // end sampleUnidirectionalKernel()

void CudaShadingContext
  ::sampleUnidirectionalScattering(const device_ptr<const CudaScatteringDistributionFunction> &f,
                                   const CudaDifferentialGeometryArray &dg,
                                   const device_ptr<const float3> &u,
                                   const size_t uStride,
                                   const device_ptr<float3> &s,
                                   const device_ptr<float3> &wo,
                                   const size_t woStride,
                                   const device_ptr<float> &pdf,
                                   const device_ptr<bool> &delta,
                                   const size_t n)
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / BLOCK_SIZE;

  AltUniParameters p = {f, dg, u, uStride, s, wo, woStride, pdf, delta};

  if(gridSize)
    sampleUnidirectionalKernel<<<gridSize,BLOCK_SIZE>>>(p);
  if(n%BLOCK_SIZE)
  {
    AltUniParameters p = {f + gridSize*BLOCK_SIZE,
                          dg + gridSize*BLOCK_SIZE,
                          stride_cast(u.get(),  gridSize*BLOCK_SIZE, uStride),
                          uStride,
                          s + gridSize+BLOCK_SIZE,
                          stride_cast(wo.get(), gridSize*BLOCK_SIZE, woStride),
                          woStride,
                          pdf + gridSize*BLOCK_SIZE,
                          delta + gridSize*BLOCK_SIZE};
    sampleUnidirectionalKernel<<<1,n%BLOCK_SIZE>>>(p);
  } // end if
} // end CudaDifferentialGeometry::sampleUnidirectionalScattering()

__global__ void evaluateBidirectionalKernel(const CudaScatteringDistributionFunction *f,
                                            const float3 *wo,
                                            const float3 *point,
                                            const float3 *tangent,
                                            const float3 *binormal,
                                            const float3 *normal,
                                            const float3 *wi,
                                            const bool *stencil,
                                            float3 *results)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(stencil[i])
  {
    results[i] = f[i].evaluate(wo[i], point[i], tangent[i], binormal[i], normal[i], wi[i]);
  } // end if
} // end evaluateBidirectionalKernel()

void CudaShadingContext
  ::evaluateBidirectionalScattering(const device_ptr<const CudaScatteringDistributionFunction> &f,
                                    const device_ptr<const float3> &wo,
                                    const CudaDifferentialGeometryArray &dg,
                                    const device_ptr<const float3> &wi,
                                    const device_ptr<const bool> &stencil,
                                    const device_ptr<float3> &results,
                                    const size_t n)
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / BLOCK_SIZE;

  if(gridSize)
    evaluateBidirectionalKernel<<<gridSize,BLOCK_SIZE>>>(f,
                                                         wo,
                                                         dg.mPoints,
                                                         dg.mTangents,
                                                         dg.mBinormals,
                                                         dg.mNormals,
                                                         wi,
                                                         stencil,
                                                         results);
  if(n%BLOCK_SIZE)
    evaluateBidirectionalKernel<<<1,n%BLOCK_SIZE>>>(f + gridSize*BLOCK_SIZE,
                                                    wo + gridSize*BLOCK_SIZE,
                                                    dg.mPoints + gridSize*BLOCK_SIZE,
                                                    dg.mTangents + gridSize*BLOCK_SIZE,
                                                    dg.mBinormals + gridSize*BLOCK_SIZE,
                                                    dg.mNormals + gridSize*BLOCK_SIZE,
                                                    wi + gridSize*BLOCK_SIZE,
                                                    stencil + gridSize*BLOCK_SIZE,
                                                    results + gridSize*BLOCK_SIZE);
} // end CudaDifferentialGeometryArray::evaluateBidirectionalScattering()

__global__ void evaluateUnidirectionalKernel(const CudaScatteringDistributionFunction *f,
                                             const float3 *wo,
                                             const float3 *point,
                                             const float3 *tangent,
                                             const float3 *binormal,
                                             const float3 *normal,
                                             const bool *stencil,
                                             float3 *results)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(stencil[i])
  {
    results[i] = f[i].evaluate(wo[i], point[i], tangent[i], binormal[i], normal[i]);
  } // end if
} // end evaluateUnidirectionalKernel()

void CudaShadingContext
  ::evaluateUnidirectionalScattering(const device_ptr<const CudaScatteringDistributionFunction> &f,
                                     const device_ptr<const float3> &wo,
                                     const CudaDifferentialGeometryArray &dg,
                                     const device_ptr<const bool> &stencil,
                                     const device_ptr<float3> &results,
                                     const size_t n)
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / BLOCK_SIZE;

  if(gridSize)
    evaluateUnidirectionalKernel<<<gridSize,BLOCK_SIZE>>>(f,
                                                          wo,
                                                          dg.mPoints,
                                                          dg.mTangents,
                                                          dg.mBinormals,
                                                          dg.mNormals,
                                                          stencil,
                                                          results);
  if(n%BLOCK_SIZE)
    evaluateUnidirectionalKernel<<<1,n%BLOCK_SIZE>>>(f + gridSize*BLOCK_SIZE,
                                                     wo + gridSize*BLOCK_SIZE,
                                                     dg.mPoints + gridSize*BLOCK_SIZE,
                                                     dg.mTangents + gridSize*BLOCK_SIZE,
                                                     dg.mBinormals + gridSize*BLOCK_SIZE,
                                                     dg.mNormals + gridSize*BLOCK_SIZE,
                                                     stencil + gridSize*BLOCK_SIZE,
                                                     results + gridSize*BLOCK_SIZE);
} // end CudaDifferentialGeometryArray::evaluateUnidirectionalScattering()

struct AltBiParameters
{
  const CudaScatteringDistributionFunction *f;
  const float3 *wo;
  int woStride;
  CudaDifferentialGeometryArray dg;
  const float3 *u;
  int uStride;
  const bool *stencil;
  float3 *s;
  float3 *wi;
  int wiStride;
  float *pdf;
  bool *delta;
  unsigned int *component;
}; // end Parameters

void __global__ sampleAltBidirectionalKernel(const AltBiParameters p)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // sample
  if(p.stencil[i])
  {
    const char *ptr = reinterpret_cast<const char*>(p.wo) + i*p.woStride;
    const float3 wo = *reinterpret_cast<const float3*>(ptr);

    ptr = reinterpret_cast<const char*>(p.u) + i*p.uStride;
    float3 u = *reinterpret_cast<const float3*>(ptr);

    char *temp = reinterpret_cast<char*>(p.wi) + i*p.wiStride;
    float3 &wi = *reinterpret_cast<float3*>(temp);

    const CudaScatteringDistributionFunction &f = p.f[i];

    f.sample(wo,
             p.dg.mPoints[i],
             p.dg.mTangents[i],
             p.dg.mBinormals[i],
             p.dg.mNormals[i],
             u.x, u.y, u.z,
             p.s[i],
             wi,
             p.pdf[i],
             p.delta[i],
             p.component[i]);
  } // end if
} // end sampleAltUnidirectionalKernel()

void CudaShadingContext
  ::sampleBidirectionalScattering(const device_ptr<const CudaScatteringDistributionFunction> &f,
                                  const device_ptr<const float3> &wo,
                                  const size_t woStride,
                                  const CudaDifferentialGeometryArray &dg,
                                  const device_ptr<const float3> &u,
                                  const size_t uStride,
                                  const device_ptr<bool> &stencil,
                                  const device_ptr<float3> &s,
                                  const device_ptr<float3> &wi,
                                  const size_t wiStride,
                                  const device_ptr<float> &pdf,
                                  const device_ptr<bool> &delta,
                                  const device_ptr<unsigned int> &component,
                                  const size_t n)
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / BLOCK_SIZE;

  AltBiParameters p = {f,
                       wo,
                       woStride,
                       dg,
                       u,
                       uStride,
                       stencil,
                       s,
                       wi,
                       wiStride,
                       pdf,
                       delta,
                       component};

  if(gridSize)
    sampleAltBidirectionalKernel<<<gridSize,BLOCK_SIZE>>>(p);
  if(n%BLOCK_SIZE)
  {
    AltBiParameters p = {f + gridSize*BLOCK_SIZE,
                         stride_cast(wo.get(), gridSize*BLOCK_SIZE, woStride),
                         woStride,
                         dg + gridSize*BLOCK_SIZE,
                         stride_cast(u.get(),  gridSize*BLOCK_SIZE, uStride),
                         uStride,
                         stencil + gridSize*BLOCK_SIZE,
                         s + gridSize*BLOCK_SIZE,
                         stride_cast(wi.get(), gridSize*BLOCK_SIZE, wiStride),
                         wiStride,
                         pdf + gridSize*BLOCK_SIZE,
                         delta + gridSize*BLOCK_SIZE,
                         component + gridSize*BLOCK_SIZE};
    sampleAltBidirectionalKernel<<<1,n%BLOCK_SIZE>>>(p);
  } // end if
} // end CudaShadingContext::sampleBidirectionalScattering()

