/*! \file CudaTriangleList.cu
 *  \author Jared Hoberock
 *  \brief Implementation of CudaTriangleList class.
 */

#include "CudaTriangleList.h"
#include "createDifferentialGeometry.h"

using namespace stdcuda;

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
  const CudaTriangleList::TriangleTable::Entry *table;
  unsigned int tableSize;
  float invTotalArea;
};

// XXX use this function while the TriangleTable is broken
__device__ unsigned int sampleTable(const CudaTriangleList::TriangleTable::Entry *table,
                                    const unsigned int tableSize,
                                    const float u,
                                    float &pdf)
{
  float q = static_cast<float>(tableSize) * u;
  unsigned int i = static_cast<unsigned int>(q);
  float u1 = q - i;
  const CudaTriangleList::TriangleTable::Entry &e = table[i];

  unsigned int result = 0;
  if(u1 < e.mDivide)
  {
    result = e.mValue;
    pdf = e.mValuePdf;
  } // end if
  else
  {
    result = e.mAlias;
    pdf = e.mAliasPdf;
  } // end else

  return result;
} // end sampleTable()

__global__ void k(const float4 *uniformFloats,
                  const Mesh m,
                  PrimitiveHandle *prims,
                  CudaDifferentialGeometry *dgs,
                  float *pdfs)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  float4 u = uniformFloats[i];

  // sample a triangle index
  // XXX fix this
  //unsigned int triIndex = m.table(u.x, pdfs[i]);

  //unsigned int triIndex = 0;
  //pdfs[i] = 1.0f;

  // sample a triangle
  float pdf = 0;
  unsigned int triIndex = sampleTable(m.table, m.tableSize, u.x, pdf);
  // XXX we should multiply by this
  //pdf *= m.invTotalArea;
  pdf = m.invTotalArea;
  pdfs[i] = pdf;

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
            &mPrimitiveHandlesDevice[0],
            //mSurfaceAreaPdf
            &mSurfaceAreaPdf.getTable()[0],
            mSurfaceAreaPdf.getTable().size(),
            1.0f / mTotalSurfaceArea
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

struct Parameters
{
  const float4 *uniformFloats;
  Mesh m;
  PrimitiveHandle *prims;
  CudaDifferentialGeometryArray dg;
  float *pdfs;
};

//__global__ void sampleSurfaceAreaKernel(const float4 *uniformFloats,
//                                        const Mesh m,
//                                        PrimitiveHandle *prims,
//                                        CudaDifferentialGeometryArray dg,
//                                        float *pdfs)
__global__ void sampleSurfaceAreaKernel(Parameters p)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  float4 u = p.uniformFloats[i];

  // sample a triangle index
  // XXX fix this
  //unsigned int triIndex = m.table(u.x, pdfs[i]);

  //unsigned int triIndex = 0;
  //pdfs[i] = 1.0f;

  // sample a triangle
  float pdf = 0;
  unsigned int triIndex = sampleTable(p.m.table, p.m.tableSize, u.x, pdf);
  // XXX we should multiply by this
  //pdf *= m.invTotalArea;
  pdf = p.m.invTotalArea;
  p.pdfs[i] = pdf;

  // get that triangle's PrimitiveHandle
  p.prims[i] = p.m.primitiveHandles[triIndex];

  // sample the isoceles right triangle
  float2 b;
  float su = sqrtf(u.y);
  b.x = 1.0f - su;
  b.y = u.z * su;

  // transform to this particular triangle
  float3 x = b.x * p.m.v0[triIndex]
           + b.y * p.m.v1[triIndex]
           + (1.0f - b.x - b.y) * p.m.v2[triIndex];

  // create DifferentialGeometry
  // set a few things ourself
  p.dg.mPoints[i] = x;
  float invSa = p.m.inverseSurfaceArea[triIndex];
  p.dg.mInverseSurfaceAreas[i] = invSa;
  p.dg.mSurfaceAreas[i] = 1.0f / invSa;

  // delegate the rest of the work
  createDifferentialGeometry(b, triIndex,
                             p.m.v0, p.m.v1, p.m.v2, p.m.n,
                             p.m.uv0, p.m.uv1, p.m.uv2,
                             p.dg.mNormals[i],
                             p.dg.mTangents[i],
                             p.dg.mBinormals[i],
                             p.dg.mParametricCoordinates[i],
                             p.dg.mDPDUs[i],
                             p.dg.mDPDVs[i],
                             p.dg.mDNDUs[i],
                             p.dg.mDNDVs[i]);
} // end k2()

void CudaTriangleList
  ::sampleSurfaceArea(const device_ptr<const float4> &u,
                      const device_ptr<PrimitiveHandle> &prims,
                      CudaDifferentialGeometryArray &dg,
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
            &mPrimitiveHandlesDevice[0],
            //mSurfaceAreaPdf
            &mSurfaceAreaPdf.getTable()[0],
            mSurfaceAreaPdf.getTable().size(),
            1.0f / mTotalSurfaceArea
           };

  Parameters p = {u, m, prims, dg, pdf};

  if(gridSize)
    //sampleSurfaceAreaKernel<<<gridSize,BLOCK_SIZE>>>(u, m, prims, dg, pdf);
    sampleSurfaceAreaKernel<<<gridSize,BLOCK_SIZE>>>(p);
  if(n%BLOCK_SIZE)
  {
    //sampleSurfaceAreaKernel<<<1,n%BLOCK_SIZE>>>(u     + gridSize*BLOCK_SIZE,
    //                                            m,
    //                                            prims + gridSize*BLOCK_SIZE,
    //                                            dg    + gridSize*BLOCK_SIZE,
    //                                            pdf   + gridSize*BLOCK_SIZE);
    Parameters p = {u + gridSize*BLOCK_SIZE,
                    m,
                    prims + gridSize*BLOCK_SIZE,
                    dg + gridSize*BLOCK_SIZE,
                    pdf + gridSize*BLOCK_SIZE};
    sampleSurfaceAreaKernel<<<1,n%BLOCK_SIZE>>>(p);
  } // end if
} // end CudaTriangleList::sampleSurfaceArea()

__global__ void sampleSurfaceAreaKernel(Parameters p,
                                        const bool *stencil)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(stencil[i])
  {
    float4 u = p.uniformFloats[i];

    // sample a triangle index
    // XXX fix this
    //unsigned int triIndex = m.table(u.x, pdfs[i]);

    //unsigned int triIndex = 0;
    //pdfs[i] = 1.0f;

    // sample a triangle
    float pdf = 0;
    unsigned int triIndex = sampleTable(p.m.table, p.m.tableSize, u.x, pdf);
    // XXX we should multiply by this
    //pdf *= m.invTotalArea;
    pdf = p.m.invTotalArea;
    p.pdfs[i] = pdf;

    // get that triangle's PrimitiveHandle
    p.prims[i] = p.m.primitiveHandles[triIndex];

    // sample the isoceles right triangle
    float2 b;
    float su = sqrtf(u.y);
    b.x = 1.0f - su;
    b.y = u.z * su;

    // transform to this particular triangle
    float3 x = b.x * p.m.v0[triIndex]
             + b.y * p.m.v1[triIndex]
             + (1.0f - b.x - b.y) * p.m.v2[triIndex];

    // create DifferentialGeometry
    // set a few things ourself
    p.dg.mPoints[i] = x;
    float invSa = p.m.inverseSurfaceArea[triIndex];
    p.dg.mInverseSurfaceAreas[i] = invSa;
    p.dg.mSurfaceAreas[i] = 1.0f / invSa;

    // delegate the rest of the work
    createDifferentialGeometry(b, triIndex,
                               p.m.v0, p.m.v1, p.m.v2, p.m.n,
                               p.m.uv0, p.m.uv1, p.m.uv2,
                               p.dg.mNormals[i],
                               p.dg.mTangents[i],
                               p.dg.mBinormals[i],
                               p.dg.mParametricCoordinates[i],
                               p.dg.mDPDUs[i],
                               p.dg.mDPDVs[i],
                               p.dg.mDNDUs[i],
                               p.dg.mDNDVs[i]);
  } // end if
} // end k2()

void CudaTriangleList
  ::sampleSurfaceArea(const device_ptr<const float4> &u,
                      const device_ptr<const bool> &stencil,
                      const device_ptr<PrimitiveHandle> &prims,
                      CudaDifferentialGeometryArray &dg,
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
            &mPrimitiveHandlesDevice[0],
            //mSurfaceAreaPdf
            &mSurfaceAreaPdf.getTable()[0],
            mSurfaceAreaPdf.getTable().size(),
            1.0f / mTotalSurfaceArea
           };

  Parameters p = {u, m, prims, dg, pdf};

  if(gridSize)
    sampleSurfaceAreaKernel<<<gridSize,BLOCK_SIZE>>>(p, stencil);
  if(n%BLOCK_SIZE)
  {
    Parameters p = {u + gridSize*BLOCK_SIZE,
                    m,
                    prims + gridSize*BLOCK_SIZE,
                    dg + gridSize*BLOCK_SIZE,
                    pdf + gridSize*BLOCK_SIZE};
    sampleSurfaceAreaKernel<<<1,n%BLOCK_SIZE>>>(p, stencil + gridSize*BLOCK_SIZE);
  } // end if
} // end CudaTriangleList::sampleSurfaceArea()

