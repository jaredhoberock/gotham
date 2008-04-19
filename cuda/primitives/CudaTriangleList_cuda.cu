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

struct Mesh
{
  const float3 *v0;
  const float3 *v1;
  const float3 *v2;
  const float3 *n;
  const float2 *uv0;
  const float2 *uv1;
  const float2 *uv2;
  const float *inverseSurfaceArea;
  const PrimitiveHandle *primitiveHandles;
  //CudaTriangleList::TriangleTable table;
};

__global__ void k(const float4 *uniformFloats,
                  const Mesh m,
                  PrimitiveHandle *prims,
                  CudaDifferentialGeometry *dgs,
                  float *pdfs)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  float4 u = uniformFloats[i];

  // sample a triangle index
  //unsigned int triIndex = m.table(u.x, pdfs[i]);
  unsigned int triIndex = 0;

  // get that triangle's PrimitiveHandle
  prims[i] = m.primitiveHandles[triIndex];

  // sample the isoceles right triangle
  float2 b;
  float su = sqrtf(u.y);
  b.x = 1.0f - su;
  b.y = u.z * su;

  // transform to this particular triangle
  float3 x = b.x * m.v0[triIndex]
           + b.y * m.v1[triIndex]
           + (1.0f - b.x - b.y) * m.v2[triIndex];

  // create DifferentialGeometry
  CudaDifferentialGeometry dg;
  createDifferentialGeometry(x, b, triIndex,
                             m.v0, m.v1, m.v2, m.n,
                             m.uv0, m.uv1, m.uv2,
                             m.inverseSurfaceArea,
                             dg);

  // write back
  dgs[i] = dg;
} // end kernel()

void CudaTriangleList
  ::sampleSurfaceArea(const device_ptr<const float4> &u,
                      const device_ptr<PrimitiveHandle> &prims,
                      const device_ptr<CudaDifferentialGeometry> &dg,
                      const device_ptr<float> &pdf,
                      const size_t n) const
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / 192;

  Mesh m = {
            &mFirstVertex[0],
            &mSecondVertex[0],
            &mThirdVertex[0],
            &mGeometricNormalDevice[0],
            &mFirstVertexParmsDevice[0],
            &mSecondVertexParmsDevice[0],
            &mThirdVertexParmsDevice[0],
            &mPrimitiveInvSurfaceAreaDevice[0],
            &mPrimitiveHandlesDevice[0]//,
            //mSurfaceAreaPdf
           };

  if(gridSize)
    k<<<gridSize,BLOCK_SIZE>>>(u, m, prims, dg, pdf);
  if(n%BLOCK_SIZE)
    k<<<1,n%BLOCK_SIZE>>>(u     + gridSize*BLOCK_SIZE,
                          m,
                          prims + gridSize*BLOCK_SIZE,
                          dg    + gridSize*BLOCK_SIZE,
                          pdf   + gridSize*BLOCK_SIZE);
} // end CudaTriangleList::sampleSurfaceArea()

