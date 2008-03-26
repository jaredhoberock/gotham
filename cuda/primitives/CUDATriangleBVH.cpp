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

void CUDATriangleBVH
  ::finalize(void)
{
  // call the parent
  Parent::finalize();

  // copy host data to device
  mMinBoundHitIndexDevice.resize(mMinBoundHitIndex.size());
  stdcuda::copy(mMinBoundHitIndex.begin(), mMinBoundHitIndex.end(), mMinBoundHitIndexDevice.begin());

  mMaxBoundMissIndexDevice.resize(mMaxBoundMissIndex.size());
  stdcuda::copy(mMaxBoundMissIndex.begin(), mMaxBoundMissIndex.end(), mMaxBoundMissIndexDevice.begin());

  mFirstVertexDominantAxisDevice.resize(mFirstVertexDominantAxis.size());
  stdcuda::copy(mFirstVertexDominantAxis.begin(), mFirstVertexDominantAxis.end(), mFirstVertexDominantAxisDevice.begin());

  createPerTriangleGeometricData();
  createScratchSpace();
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
  ::intersect(Ray *rays,
              Intersection *intersections,
              int *stencil,
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
  cudaRayTriangleBVHIntersection(NULL_NODE,
                                 mRootIndex,
                                 &mRayOriginsAndMinT[0],
                                 &mRayDirectionsAndMaxT[0],
                                 &mMinBoundHitIndexDevice[0],
                                 &mMaxBoundMissIndexDevice[0],
                                 &mFirstVertexDominantAxisDevice[0],
                                 &mDeviceStencil[0],
                                 &mTimeBarycentricsAndTriangleIndex[0],
                                 n);

  // transform results into Intersections
  createIntersections(&mRayOriginsAndMinT[0],
                      &mRayDirectionsAndMaxT[0],
                      &mTimeBarycentricsAndTriangleIndex[0],
                      &mDeviceStencil[0],
                      intersections,
                      n);

  // copy deviceStencil to host
  stdcuda::copy(mDeviceStencil.begin(),
                mDeviceStencil.end(),
                stencil);
} // end CUDATriangleBVH::intersect()

void CUDATriangleBVH
  ::createIntersections(stdcuda::const_device_pointer<float4> rayOriginsAndMinT,
                        stdcuda::const_device_pointer<float4> rayDirectionsAndMaxT,
                        stdcuda::const_device_pointer<float4> timeBarycentricsAndTriangleIndex,
                        stdcuda::const_device_pointer<int> stencil,
                        Intersection *intersections,
                        const size_t n) const
{
  stdcuda::vector_dev<CudaIntersection> deviceIntersections(n);
  cudaCreateIntersections(rayOriginsAndMinT, rayDirectionsAndMaxT,
                          timeBarycentricsAndTriangleIndex,
                          &mGeometricNormalDevice[0],
                          &mFirstVertexParmsDevice[0],
                          &mSecondVertexParmsDevice[0],
                          &mThirdVertexParmsDevice[0],
                          &mPrimitiveInvSurfaceAreaDevice[0],
                          &mPrimitiveHandlesDevice[0],
                          stencil,
                          &deviceIntersections[0],
                          n);

  //std::cerr << "back from kernel" << std::endl;
  //std::cerr << "error string: " << cudaGetErrorString(cudaGetLastError()) << std::endl;

  // XXX this is a bit of a hack, but it will work (now that Intersection is padded)
  CudaIntersection *hostPtr = reinterpret_cast<CudaIntersection*>(intersections);
  stdcuda::copy(deviceIntersections.begin(),
                deviceIntersections.end(),
                hostPtr);
} // end CUDATriangleBVH::createIntersections()

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

  // XXX perf this copy up
  for(size_t i = 0; i != n; ++i)
  {
    // figure out which triangle of which mesh we are
    // interested in
    const Triangle &globalTri = mTriangles[i];
    PrimitiveHandle prim = globalTri.mPrimitiveHandle;
    size_t localTriIndex = globalTri.mTriangleIndex;

    const SurfacePrimitive *sp = static_cast<const SurfacePrimitive*>((*this)[prim].get());

    const Mesh *mesh = static_cast<const Mesh *>(sp->getSurface());
    const Mesh::PointList &points = mesh->getPoints();
    const Mesh::Triangle &tri = mesh->getTriangles()[localTriIndex];

    Vector e1 = points[tri[1]] - points[tri[0]];
    Vector e2 = points[tri[2]] - points[tri[0]];
    Vector n = e1.cross(e2).normalize();

    ParametricCoordinates uv0, uv1, uv2;
    mesh->getParametricCoordinates(tri, uv0, uv1, uv2);

    // XXX this will be slow as balls
    mGeometricNormalDevice[i]         = make_float3(n[0], n[1], n[2]);
    mFirstVertexParmsDevice[i]        = make_float2(uv0[0], uv0[1]);
    mSecondVertexParmsDevice[i]       = make_float2(uv1[0], uv1[1]);
    mThirdVertexParmsDevice[i]        = make_float2(uv2[0], uv2[1]);
    mPrimitiveHandlesDevice[i]        = prim;
    mPrimitiveInvSurfaceAreaDevice[i] = sp->getInverseSurfaceArea();
  } // end for i
} // end CUDATriangleBVH::createPerTriangleGeometricData()

void CUDATriangleBVH
  ::setWorkBatchSize(const size_t b)
{
  mWorkBatchSize = b;
} // end CUDATriangleBVH::setWorkBatchSize()

