/*! \file CudaTriangleBVH.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of CudaTriangleBVH class.
 */

#include "CudaTriangleBVH.h"
#include <vector_functions.h>
#include <stdcuda/stdcuda.h>
#include "cudaRayTriangleBVHIntersection.h"
#include "cudaCreateIntersections.h"
#include <stdcuda/cuda_algorithm.h>
#include "../../primitives/SurfacePrimitive.h"
#include "CudaIntersection.h"
#include <bittricks/bittricks.h>

using namespace stdcuda;

void CudaTriangleBVH
  ::finalize(void)
{
  // call the parent
  Parent0::finalize();
  
  // copy host data to device
  mMinBoundHitIndexDevice.resize(mMinBoundHitIndex.size());
  stdcuda::copy(mMinBoundHitIndex.begin(), mMinBoundHitIndex.end(), mMinBoundHitIndexDevice.begin());

  mMaxBoundMissIndexDevice.resize(mMaxBoundMissIndex.size());
  stdcuda::copy(mMaxBoundMissIndex.begin(), mMaxBoundMissIndex.end(), mMaxBoundMissIndexDevice.begin());

  mFirstVertexDominantAxisDevice.resize(mFirstVertexDominantAxis.size());
  stdcuda::copy(mFirstVertexDominantAxis.begin(), mFirstVertexDominantAxis.end(), mFirstVertexDominantAxisDevice.begin());

  createPerTriangleGeometricData();
  createScratchSpace();
  createPrimitiveHandleToMaterialHandleMap();

  // call the other parent
  Parent2::finalize();
} // end CudaTriangleBVH::finalize()

void CudaTriangleBVH
  ::createScratchSpace(void)
{
  mTimeBarycentricsAndTriangleIndex.resize(mWorkBatchSize);
} // end CudaTriangleBVH::createScratchSpace()

void CudaTriangleBVH
  ::intersect(const device_ptr<const float3> &origins,
              const device_ptr<const float3> &directions,
              const device_ptr<const float2> &intervals,
              const device_ptr<const bool> &stencil,
              const device_ptr<bool> &results,
              const size_t n) const
{
  // we can only do as much work as we have preallocated space
  size_t m = std::min(n, mWorkBatchSize);

  // perform intersection
  cudaShadowRayTriangleBVHIntersectionWithStencil(NULL_NODE,
                                                  mRootIndex,
                                                  origins,
                                                  directions,
                                                  intervals,
                                                  &mMinBoundHitIndexDevice[0],
                                                  &mMaxBoundMissIndexDevice[0],
                                                  &mFirstVertexDominantAxisDevice[0],
                                                  stencil,
                                                  results,
                                                  m);
} // end CudaTriangleBVH::intersect()

void CudaTriangleBVH
  ::intersect(const device_ptr<const float3> &origins,
              const device_ptr<const float3> &directions,
              const float2 &interval,
              const device_ptr<const bool> &stencil,
              const device_ptr<bool> &results,
              const size_t n) const
{
  // we can only do as much work as we have preallocated space
  size_t m = std::min(n, mWorkBatchSize);

  // perform intersection
  cudaShadowRayTriangleBVHIntersectionWithStencil(NULL_NODE,
                                                  mRootIndex,
                                                  origins,
                                                  directions,
                                                  interval,
                                                  &mMinBoundHitIndexDevice[0],
                                                  &mMaxBoundMissIndexDevice[0],
                                                  &mFirstVertexDominantAxisDevice[0],
                                                  stencil,
                                                  results,
                                                  m);
} // end CudaTriangleBVH::intersect()

void CudaTriangleBVH
  ::intersect(const stdcuda::device_ptr<const float3> &origins,
              const stdcuda::device_ptr<const float3> &directions,
              const stdcuda::device_ptr<const float2> &intervals,
              CudaDifferentialGeometryArray &dg,
              const stdcuda::device_ptr<PrimitiveHandle> &hitPrims,
              const stdcuda::device_ptr<bool> &stencil,
              const size_t n) const
{
  // we can only do as much work as we have preallocated space
  size_t m = std::min(n, mWorkBatchSize);

  // perform intersection
  cudaRayTriangleBVHIntersection(NULL_NODE,
                                 mRootIndex,
                                 origins,
                                 directions,
                                 intervals,
                                 &mMinBoundHitIndexDevice[0],
                                 &mMaxBoundMissIndexDevice[0],
                                 &mFirstVertexDominantAxisDevice[0],
                                 stencil,
                                 &mTimeBarycentricsAndTriangleIndex[0],
                                 m);

  // transform intermediate data into intersection information
  createIntersectionData(origins, directions,
                         &mTimeBarycentricsAndTriangleIndex[0],
                         &mGeometricNormalDevice[0],
                         &mFirstVertex[0],
                         &mSecondVertex[0],
                         &mThirdVertex[0],
                         &mFirstVertexParmsDevice[0],
                         &mSecondVertexParmsDevice[0],
                         &mThirdVertexParmsDevice[0],
                         &mPrimitiveInvSurfaceAreaDevice[0],
                         &mPrimitiveHandlesDevice[0],
                         stencil,
                         dg,
                         hitPrims,
                         m);
} // end CudaTriangleBVH::intersect()

void CudaTriangleBVH
  ::intersect(const stdcuda::device_ptr<const float3> &origins,
              const stdcuda::device_ptr<const float3> &directions,
              const float2 &interval,
              CudaDifferentialGeometryArray &dg,
              const stdcuda::device_ptr<PrimitiveHandle> &hitPrims,
              const stdcuda::device_ptr<bool> &stencil,
              const size_t n) const
{
  // we can only do as much work as we have preallocated space
  size_t m = std::min(n, mWorkBatchSize);

  // perform intersection
  cudaRayTriangleBVHIntersection(NULL_NODE,
                                 mRootIndex,
                                 origins,
                                 directions,
                                 interval,
                                 &mMinBoundHitIndexDevice[0],
                                 &mMaxBoundMissIndexDevice[0],
                                 &mFirstVertexDominantAxisDevice[0],
                                 stencil,
                                 &mTimeBarycentricsAndTriangleIndex[0],
                                 m);

  // transform intermediate data into intersection information
  createIntersectionData(origins, directions,
                         &mTimeBarycentricsAndTriangleIndex[0],
                         &mGeometricNormalDevice[0],
                         &mFirstVertex[0],
                         &mSecondVertex[0],
                         &mThirdVertex[0],
                         &mFirstVertexParmsDevice[0],
                         &mSecondVertexParmsDevice[0],
                         &mThirdVertexParmsDevice[0],
                         &mPrimitiveInvSurfaceAreaDevice[0],
                         &mPrimitiveHandlesDevice[0],
                         stencil,
                         dg,
                         hitPrims,
                         m);
} // end CudaTriangleBVH::intersect()

void CudaTriangleBVH
  ::createPerTriangleGeometricData(void)
{
  size_t n = mFirstVertexDominantAxisDevice.size();

  mGeometricNormalDevice.resize(n);
  mFirstVertexParmsDevice.resize(n);
  mSecondVertexParmsDevice.resize(n);
  mThirdVertexParmsDevice.resize(n);
  mPrimitiveInvSurfaceAreaDevice.resize(n);
  mPrimitiveHandlesDevice.resize(n);

  Parent2::mFirstVertex.resize(n);
  Parent2::mSecondVertex.resize(n);
  Parent2::mThirdVertex.resize(n);

  // compute into temporary host-side arrays
  std::vector<float3> geometricNormal(n);
  std::vector<float3> firstVertex(n);
  std::vector<float3> secondVertex(n);
  std::vector<float3> thirdVertex(n);
  std::vector<float2> firstVertexParms(n);
  std::vector<float2> secondVertexParms(n);
  std::vector<float2> thirdVertexParms(n);
  std::vector<PrimitiveHandle> primitiveHandles(n);
  std::vector<float> primitiveInvSurfaceArea(n);
  for(size_t i = 0; i != n; ++i)
  {
    // figure out which triangle of which mesh we are
    // interested in
    const Triangle &globalTri = mTriangles[i];
    size_t localTriIndex = globalTri.mTriangleIndex;
    const SurfacePrimitive *sp = globalTri.mPrimitive;

    const Mesh *mesh = static_cast<const Mesh *>(sp->getSurface());
    const Mesh::PointList &points = mesh->getPoints();
    const Mesh::Triangle &tri = mesh->getTriangles()[localTriIndex];

    Point v0 = points[tri[0]];
    Point v1 = points[tri[1]];
    Point v2 = points[tri[2]];

    Vector e1 = v1 - v0;
    Vector e2 = v2 - v0;
    Vector n = e1.cross(e2).normalize();

    ParametricCoordinates uv0, uv1, uv2;
    mesh->getParametricCoordinates(tri, uv0, uv1, uv2);

    geometricNormal[i]         = make_float3(n[0], n[1], n[2]);
    firstVertexParms[i]        = make_float2(uv0[0], uv0[1]);
    secondVertexParms[i]       = make_float2(uv1[0], uv1[1]);
    thirdVertexParms[i]        = make_float2(uv2[0], uv2[1]);
    primitiveHandles[i]        = sp->getPrimitiveHandle();
    primitiveInvSurfaceArea[i] = sp->getInverseSurfaceArea();

    firstVertex[i]  = make_float3(v0[0], v0[1], v0[2]);
    secondVertex[i] = make_float3(v1[0], v1[1], v1[2]);
    thirdVertex[i]  = make_float3(v2[0], v2[1], v2[2]);
  } // end for i

  // copy everything
  stdcuda::copy(geometricNormal.begin(), geometricNormal.end(), mGeometricNormalDevice.begin());
  stdcuda::copy(firstVertexParms.begin(), firstVertexParms.end(), mFirstVertexParmsDevice.begin());
  stdcuda::copy(secondVertexParms.begin(), secondVertexParms.end(), mSecondVertexParmsDevice.begin());
  stdcuda::copy(thirdVertexParms.begin(), thirdVertexParms.end(), mThirdVertexParmsDevice.begin());
  stdcuda::copy(primitiveHandles.begin(), primitiveHandles.end(), mPrimitiveHandlesDevice.begin());
  stdcuda::copy(primitiveInvSurfaceArea.begin(), primitiveInvSurfaceArea.end(), mPrimitiveInvSurfaceAreaDevice.begin());
  stdcuda::copy(firstVertex.begin(), firstVertex.end(), Parent2::mFirstVertex.begin());
  stdcuda::copy(secondVertex.begin(), secondVertex.end(), Parent2::mSecondVertex.begin());
  stdcuda::copy(thirdVertex.begin(), thirdVertex.end(), Parent2::mThirdVertex.begin());
} // end CudaTriangleBVH::createPerTriangleGeometricData()

void CudaTriangleBVH
  ::setWorkBatchSize(const size_t b)
{
  mWorkBatchSize = b;
} // end CudaTriangleBVH::setWorkBatchSize()

void CudaTriangleBVH
  ::createPrimitiveHandleToMaterialHandleMap(void)
{
  Parent2::mPrimitiveHandleToMaterialHandle.resize(size());

  // compute into a temporary host-side array
  std::vector<MaterialHandle> temp(size());

  PrimitiveHandle h = 0;
  for(const_iterator prim = begin();
      prim != end();
      ++prim, ++h)
  {
    const SurfacePrimitive *sp = dynamic_cast<const SurfacePrimitive*>(prim->get());
    //Parent2::mPrimitiveHandleToMaterialHandle[h] = sp->getMaterial();
    temp[h] = sp->getMaterial();
  } // end for i

  // copy
  stdcuda::copy(temp.begin(), temp.end(), Parent2::mPrimitiveHandleToMaterialHandle.begin());
} // end CudaTriangleBVH::createPrimitiveHandleToMaterialHandleMap()

