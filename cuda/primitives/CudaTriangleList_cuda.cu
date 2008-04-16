/*! \file CudaTriangleList.cu
 *  \author Jared Hoberock
 *  \brief Implementation of CudaTriangleList class.
 */

#include "CudaTriangleList.h"
#include "createDifferentialGeometry.h"

using namespace stdcuda;

struct Parameters
{
  const float4 *u;
  PrimitiveHandle *prims;
  CudaDifferentialGeometry *dg;
  float *pdf;
  CudaTriangleList::TriangleTable table;
  const float3 *v0;
  const float3 *v1;
  const float3 *v2;
  const float3 *n;
  const float2 *uv0;
  const float2 *uv1;
  const float2 *uv2;
  const float  *inverseSurfaceArea;
  const PrimitiveHandle *primitiveHandles;
};

__global__ void kernel(const Parameters p)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  float4 u = p.u[i];

  // sample a triangle index
  unsigned int triIndex = p.table(u.x, p.pdf[i]);

  // get that triangle's PrimitiveHandle
  p.prims[i] = p.primitiveHandles[triIndex];

  // sample the isoceles right triangle
  float2 b;
  float su = sqrtf(u.y);
  b.x = 1.0f - su;
  b.y = u.z * su;

  // transform to this particular triangle
  float3 x = b.x * p.v0[triIndex]
           + b.y * p.v1[triIndex]
           + (1.0f - b.x - b.y) * p.v2[triIndex];

  // create DifferentialGeometry
  CudaDifferentialGeometry dg;
  createDifferentialGeometry(x, b, triIndex,
                             p.v0, p.v1, p.v2, p.n,
                             p.uv0, p.uv1, p.uv2,
                             p.inverseSurfaceArea,
                             dg);

  // write back
  p.dg[i] = dg;
} // end kernel()

void CudaTriangleList
  ::sampleSurfaceArea(const device_ptr<const float4> &u,
                      const device_ptr<PrimitiveHandle> &prims,
                      const device_ptr<CudaDifferentialGeometry> &dg,
                      const device_ptr<float> &pdf,
                      const size_t n) const
{
  dim3 grid(1,1,1);
  dim3 block(n,1,1);

  Parameters p = {u,
                  prims,
                  dg,
                  pdf,
                  mSurfaceAreaPdf,
                  &mFirstVertex[0],
                  &mSecondVertex[0],
                  &mThirdVertex[0],
                  &mGeometricNormalDevice[0],
                  &mFirstVertexParmsDevice[0],
                  &mSecondVertexParmsDevice[0],
                  &mThirdVertexParmsDevice[0],
                  &mPrimitiveInvSurfaceAreaDevice[0],
                  &mPrimitiveHandlesDevice[0]};

  kernel<<<grid,block>>>(p);
} // end CudaTriangleList::sampleSurfaceArea()

