/*! \file CUDATriangleBVH.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of CUDATriangleBVH class.
 */

#include "CUDATriangleBVH.h"
#include <vector_functions.h>
#include <stdcuda/stdcuda.h>
#include "cudaRayTriangleBVHIntersection.h"
#include "cudaCreateIntersections.h"
#include <stdcuda/cuda_algorithm.h>
#include "../../primitives/SurfacePrimitive.h"
#include "CudaIntersection.h"

using namespace stdcuda;

void CUDATriangleBVH
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
} // end CUDATriangleBVH::finalize()

void CUDATriangleBVH
  ::createScratchSpace(void)
{
  mRayOriginsAndMinT.resize(mWorkBatchSize);
  mRayDirectionsAndMaxT.resize(mWorkBatchSize);
  mDeviceStencil.resize(mWorkBatchSize);
  mTimeBarycentricsAndTriangleIndex.resize(mWorkBatchSize);
} // end CUDATriangleBVH::createScratchSpace()

void CUDATriangleBVH
  ::intersect(const device_ptr<const float4> &originsAndMinT,
              const device_ptr<const float4> &directionsAndMaxT,
              const device_ptr<const bool> &stencil,
              const device_ptr<bool> &results,
              const size_t n) const
{
  // we can only do as much work as we have preallocated space
  size_t m = std::min(n, mWorkBatchSize);

  // perform intersection
  cudaShadowRayTriangleBVHIntersectionWithStencil(NULL_NODE,
                                                  mRootIndex,
                                                  originsAndMinT,
                                                  directionsAndMaxT,
                                                  &mMinBoundHitIndexDevice[0],
                                                  &mMinBoundHitIndexDevice[0],
                                                  &mFirstVertexDominantAxisDevice[0],
                                                  stencil,
                                                  results,
                                                  m);
} // end CUDATriangleBVH::intersect()

void CUDATriangleBVH
    ::intersect(stdcuda::device_ptr<const float4> originsAndMinT,
                stdcuda::device_ptr<const float4> directionsAndMaxT,
                stdcuda::device_ptr<CudaIntersection> intersections,
                stdcuda::device_ptr<bool> stencil,
                const size_t n) const
{
  // we can only do as much work as we have preallocated space
  size_t m = std::min(n, mWorkBatchSize);

  // perform intersection
  cudaRayTriangleBVHIntersection(NULL_NODE,
                                 mRootIndex,
                                 originsAndMinT,
                                 directionsAndMaxT,
                                 &mMinBoundHitIndexDevice[0],
                                 &mMaxBoundMissIndexDevice[0],
                                 &mFirstVertexDominantAxisDevice[0],
                                 stencil,
                                 &mTimeBarycentricsAndTriangleIndex[0],
                                 m);

  // transform intermediate data into CudaIntersections
  cudaCreateIntersections(originsAndMinT, directionsAndMaxT,
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
                          intersections,
                          m);
} // end CUDATriangleBVH::intersect()

void CUDATriangleBVH
  ::intersect(Ray *rays,
              Intersection *intersections,
              bool *stencil,
              size_t n) const
{
  // we can only do as much work as we have preallocated space
  n = std::min(n, mWorkBatchSize);

  // XXX this will be painfully slow
  for(size_t i = 0; i != n; ++i)
  {
    const Ray &r = rays[i];
    mRayOriginsAndMinT[i] = make_float4(r.getAnchor()[0],
                                        r.getAnchor()[1],
                                        r.getAnchor()[2],
                                        r.getInterval()[0]);

    mRayDirectionsAndMaxT[i] = make_float4(r.getDirection()[0],
                                           r.getDirection()[1],
                                           r.getDirection()[2],
                                           r.getInterval()[1]);
  } // end for i

  // copy to the device stencil
  stdcuda::copy(stencil, stencil + n, mDeviceStencil.begin());

  // perform intersection
  stdcuda::vector_dev<CudaIntersection> deviceIntersections(n);
  intersect(&mRayOriginsAndMinT[0],
            &mRayDirectionsAndMaxT[0],
            &deviceIntersections[0],
            &mDeviceStencil[0],
            n);

  // XXX this is a bit of a hack, but it will work (now that Intersection is padded)
  CudaIntersection *hostPtr = reinterpret_cast<CudaIntersection*>(intersections);
  stdcuda::copy(deviceIntersections.begin(),
                deviceIntersections.end(),
                hostPtr);

  // copy deviceStencil to host
  stdcuda::copy(mDeviceStencil.begin(),
                mDeviceStencil.end(),
                stencil);
} // end CUDATriangleBVH::intersect()

void CUDATriangleBVH
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

  // XXX perf this copy up
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

    // XXX this will be slow as balls
    mGeometricNormalDevice[i]         = make_float3(n[0], n[1], n[2]);
    mFirstVertexParmsDevice[i]        = make_float2(uv0[0], uv0[1]);
    mSecondVertexParmsDevice[i]       = make_float2(uv1[0], uv1[1]);
    mThirdVertexParmsDevice[i]        = make_float2(uv2[0], uv2[1]);
    mPrimitiveHandlesDevice[i]        = sp->getPrimitiveHandle();
    mPrimitiveInvSurfaceAreaDevice[i] = sp->getInverseSurfaceArea();

    Parent2::mFirstVertex[i]  = make_float3(v0[0], v0[1], v0[2]);
    Parent2::mSecondVertex[i] = make_float3(v1[0], v1[1], v1[2]);
    Parent2::mThirdVertex[i]  = make_float3(v2[0], v2[1], v2[2]);
  } // end for i
} // end CUDATriangleBVH::createPerTriangleGeometricData()

void CUDATriangleBVH
  ::setWorkBatchSize(const size_t b)
{
  mWorkBatchSize = b;
} // end CUDATriangleBVH::setWorkBatchSize()

void CUDATriangleBVH
  ::createPrimitiveHandleToMaterialHandleMap(void)
{
  Parent2::mPrimitiveHandleToMaterialHandle.resize(size());

  PrimitiveHandle h = 0;
  for(const_iterator prim = begin();
      prim != end();
      ++prim, ++h)
  {
    const SurfacePrimitive *sp = dynamic_cast<const SurfacePrimitive*>(prim->get());
    Parent2::mPrimitiveHandleToMaterialHandle[h] = sp->getMaterial();
  } // end for i
} // end CUDATriangleBVH::createPrimitiveHandleToMaterialHandleMap()

