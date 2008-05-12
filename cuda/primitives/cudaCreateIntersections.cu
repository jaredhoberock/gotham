/*! \file cudaCreateIntersections.cu
 *  \author Jared Hoberock
 *  \brief Implementation of cudaCreateIntersections function.
 */

#include "cudaCreateIntersections.h"
#include "../geometry/CudaDifferentialGeometry.h"
#include "createDifferentialGeometry.h"
#include <stdcuda/vector_math.h>
#include <stdio.h>

struct Parameters
{
  const float4 *o;
  const float4 *d;
  const float4 *hitTimeBarycentricsTriangleIndex;
  const float3 *n;
  const float3 *v0;
  const float3 *v1;
  const float3 *v2;
  const float2 *uv0;
  const float2 *uv1;
  const float2 *uv2;
  const float  *inverseSurfaceArea;
  const PrimitiveHandle *primitiveHandles;
  const bool   *stencil;
  CudaIntersection *results;
};

// XXX we have to give these parameters stupid short names because of this bug
// http://forums.nvidia.com/index.php?showtopic=53767
// TODO fix this after the next CUDA release
__global__ void k(const Parameters p)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(p.stencil[i])
  {
    float4 org = p.o[i];
    float4 dir = p.d[i];
    float4 result = p.hitTimeBarycentricsTriangleIndex[i];

    // compute hit point
    float3 hitPoint = make_float3(org.x,org.y,org.z) + result.x * make_float3(dir.x,dir.y,dir.z);

    // compute triangle index
    uint triIndex = __float_as_int(result.w);

    CudaDifferentialGeometry dg;
    createDifferentialGeometry(hitPoint, make_float2(result.y,result.z),
                               triIndex,
                               p.v0, p.v1, p.v2, p.n,
                               p.uv0, p.uv1, p.uv2,
                               p.inverseSurfaceArea,
                               dg);

    // set the DifferentialGeometry
    p.results[i].setDifferentialGeometry(dg);

    // set the primitive handle
    //     in the intersection
    p.results[i].setPrimitive(p.primitiveHandles[triIndex]);
  } // end if
} // end createIntersectionsKernel()

void cudaCreateIntersections(const float4 *rayOriginsAndMinT,
                             const float4 *rayDirectionsAndMaxT,
                             const float4 *timeBarycentricsAndTriangleIndex,
                             const float3 *geometricNormals,
                             const float3 *firstVertex,
                             const float3 *secondVertex,
                             const float3 *thirdVertex,
                             const float2 *firstVertexParms,
                             const float2 *secondVertexParms,
                             const float2 *thirdVertexParms,
                             const float  *invSurfaceArea,
                             const PrimitiveHandle *primHandles,
                             const bool *stencil,
                             CudaIntersection *intersections,
                             const size_t n)
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / 192;

  Parameters p = {rayOriginsAndMinT,
                  rayDirectionsAndMaxT,
                  timeBarycentricsAndTriangleIndex,
                  geometricNormals,
                  firstVertex,
                  secondVertex,
                  thirdVertex,
                  firstVertexParms,
                  secondVertexParms,
                  thirdVertexParms,
                  invSurfaceArea,
                  primHandles,
                  stencil,
                  intersections};

  if(gridSize)
    k<<<gridSize,BLOCK_SIZE>>>(p);
  if(n%BLOCK_SIZE)
  {
    p.o += gridSize*BLOCK_SIZE;
    p.d += gridSize*BLOCK_SIZE;
    p.hitTimeBarycentricsTriangleIndex += gridSize*BLOCK_SIZE;
    p.stencil += gridSize*BLOCK_SIZE;
    p.results += gridSize*BLOCK_SIZE;

    k<<<1,n%BLOCK_SIZE>>>(p);
  } // end if
} // end cudaCreateIntersections()

struct AltParameters
{
  const float4 *o;
  const float4 *d;
  const float4 *hitTimeBarycentricsTriangleIndex;
  const float3 *n;
  const float3 *v0;
  const float3 *v1;
  const float3 *v2;
  const float2 *uv0;
  const float2 *uv1;
  const float2 *uv2;
  const float  *inverseSurfaceArea;
  const PrimitiveHandle *primitiveHandles;
  const bool   *stencil;
  CudaDifferentialGeometryArray dg;
  PrimitiveHandle *hitPrims;
}; // end AltParameters

// XXX we have to give these parameters stupid short names because of this bug
// http://forums.nvidia.com/index.php?showtopic=53767
// TODO fix this after the next CUDA release
__global__ void k2(const AltParameters p)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(p.stencil[i])
  {
    float4 org = p.o[i];
    float4 dir = p.d[i];
    float4 result = p.hitTimeBarycentricsTriangleIndex[i];

    // compute hit point
    float3 hitPoint = make_float3(org.x,org.y,org.z) + result.x * make_float3(dir.x,dir.y,dir.z);

    // compute triangle index
    uint triIndex = __float_as_int(result.w);

    // set a few things ourself
    p.dg.mPoints[i] = hitPoint;
    float invSa = p.inverseSurfaceArea[triIndex];
    p.dg.mInverseSurfaceAreas[i] = invSa;
    p.dg.mSurfaceAreas[i] = 1.0f / invSa;

    // create the rest of the differential geometry
    createDifferentialGeometry(make_float2(result.y,result.z),
                               triIndex,
                               p.v0, p.v1, p.v2, p.n,
                               p.uv0, p.uv1, p.uv2,
                               p.dg.mNormals[i],
                               p.dg.mTangents[i],
                               p.dg.mBinormals[i],
                               p.dg.mParametricCoordinates[i],
                               p.dg.mDPDUs[i],
                               p.dg.mDPDVs[i],
                               p.dg.mDNDUs[i],
                               p.dg.mDNDVs[i]);

    // set the primitive handle
    //     in the intersection
    p.hitPrims[i] = p.primitiveHandles[triIndex];
  } // end if
} // end createIntersectionDataKernel()

void createIntersectionData(const float4 *rayOriginsAndMinT,
                            const float4 *rayDirectionsAndMaxT,
                            const float4 *timeBarycentricsAndTriangleIndex,
                            const float3 *geometricNormals,
                            const float3 *firstVertex,
                            const float3 *secondVertex,
                            const float3 *thirdVertex,
                            const float2 *firstVertexParms,
                            const float2 *secondVertexParms,
                            const float2 *thirdVertexParms,
                            const float  *invSurfaceArea,
                            const PrimitiveHandle *primHandles,
                            const bool *stencil,
                            CudaDifferentialGeometryArray &dg,
                            PrimitiveHandle *hitPrims,
                            const size_t n)
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / 192;

  AltParameters p = {rayOriginsAndMinT,
                     rayDirectionsAndMaxT,
                     timeBarycentricsAndTriangleIndex,
                     geometricNormals,
                     firstVertex,
                     secondVertex,
                     thirdVertex,
                     firstVertexParms,
                     secondVertexParms,
                     thirdVertexParms,
                     invSurfaceArea,
                     primHandles,
                     stencil,
                     dg,
                     hitPrims};

  if(gridSize)
    k2<<<gridSize,BLOCK_SIZE>>>(p);
  if(n%BLOCK_SIZE)
  {
    p.o += gridSize*BLOCK_SIZE;
    p.d += gridSize*BLOCK_SIZE;
    p.hitTimeBarycentricsTriangleIndex += gridSize*BLOCK_SIZE;
    p.stencil += gridSize*BLOCK_SIZE;

    p.hitPrims += gridSize*BLOCK_SIZE;
    p.dg += gridSize*BLOCK_SIZE;

    k2<<<1,n%BLOCK_SIZE>>>(p);
  } // end if
} // end cudaCreateIntersectionData()

struct AltAltParameters
{
  const float3 *o;
  const float3 *d;
  const float4 *hitTimeBarycentricsTriangleIndex;
  const float3 *n;
  const float3 *v0;
  const float3 *v1;
  const float3 *v2;
  const float2 *uv0;
  const float2 *uv1;
  const float2 *uv2;
  const float  *inverseSurfaceArea;
  const PrimitiveHandle *primitiveHandles;
  const bool   *stencil;
  CudaDifferentialGeometryArray dg;
  PrimitiveHandle *hitPrims;
}; // end AltParameters

// XXX we have to give these parameters stupid short names because of this bug
// http://forums.nvidia.com/index.php?showtopic=53767
// TODO fix this after the next CUDA release
__global__ void k3(const AltAltParameters p)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(p.stencil[i])
  {
    float3 org = p.o[i];
    float3 dir = p.d[i];
    float4 result = p.hitTimeBarycentricsTriangleIndex[i];

    // compute hit point
    float3 hitPoint = org + result.x * dir;

    // compute triangle index
    uint triIndex = __float_as_int(result.w);

    // set a few things ourself
    p.dg.mPoints[i] = hitPoint;
    float invSa = p.inverseSurfaceArea[triIndex];
    p.dg.mInverseSurfaceAreas[i] = invSa;
    p.dg.mSurfaceAreas[i] = 1.0f / invSa;

    // create the rest of the differential geometry
    createDifferentialGeometry(make_float2(result.y,result.z),
                               triIndex,
                               p.v0, p.v1, p.v2, p.n,
                               p.uv0, p.uv1, p.uv2,
                               p.dg.mNormals[i],
                               p.dg.mTangents[i],
                               p.dg.mBinormals[i],
                               p.dg.mParametricCoordinates[i],
                               p.dg.mDPDUs[i],
                               p.dg.mDPDVs[i],
                               p.dg.mDNDUs[i],
                               p.dg.mDNDVs[i]);

    // set the primitive handle
    //     in the intersection
    p.hitPrims[i] = p.primitiveHandles[triIndex];
  } // end if
} // end createIntersectionDataKernel()

void createIntersectionData(const float3 *origins,
                            const float3 *directions,
                            const float4 *timeBarycentricsAndTriangleIndex,
                            const float3 *geometricNormals,
                            const float3 *firstVertex,
                            const float3 *secondVertex,
                            const float3 *thirdVertex,
                            const float2 *firstVertexParms,
                            const float2 *secondVertexParms,
                            const float2 *thirdVertexParms,
                            const float  *invSurfaceArea,
                            const PrimitiveHandle *primHandles,
                            const bool *stencil,
                            CudaDifferentialGeometryArray &dg,
                            PrimitiveHandle *hitPrims,
                            const size_t n)
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / 192;

  AltAltParameters p = {origins,
                        directions,
                        timeBarycentricsAndTriangleIndex,
                        geometricNormals,
                        firstVertex,
                        secondVertex,
                        thirdVertex,
                        firstVertexParms,
                        secondVertexParms,
                        thirdVertexParms,
                        invSurfaceArea,
                        primHandles,
                        stencil,
                        dg,
                        hitPrims};

  if(gridSize)
    k3<<<gridSize,BLOCK_SIZE>>>(p);
  if(n%BLOCK_SIZE)
  {
    p.o += gridSize*BLOCK_SIZE;
    p.d += gridSize*BLOCK_SIZE;
    p.hitTimeBarycentricsTriangleIndex += gridSize*BLOCK_SIZE;
    p.stencil += gridSize*BLOCK_SIZE;

    p.hitPrims += gridSize*BLOCK_SIZE;
    p.dg += gridSize*BLOCK_SIZE;

    k3<<<1,n%BLOCK_SIZE>>>(p);
  } // end if
} // end cudaCreateIntersectionData()

