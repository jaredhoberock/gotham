/*! \file CUDATriangleBVH.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of CUDATriangleBVH class.
 */

#include "CUDATriangleBVH.h"
#include <vector_functions.h>
#include <stdcuda/stdcuda.h>
#include "cudaRayTriangleBVHIntersection.h"
#include <stdcuda/cuda_algorithm.h>

std::ostream &operator<<(std::ostream &os, const float4 &v)
{
  os << v.x << " " << v.y << " " << v.z << " " << v.w;
  return os;
}

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
} // end CUDATriangleBVH::finalize()

void CUDATriangleBVH
  ::intersect(Ray *rays,
              Intersection *intersections,
              int *stencil,
              const size_t n) const
{
  stdcuda::vector_dev<float4> rayOriginsAndMinT(n);
  stdcuda::vector_dev<float4> rayDirectionsAndMaxT(n);
  stdcuda::vector_dev<int>    deviceStencil(n);

  for(size_t i = 0; i != n; ++i)
  {
    const Ray &r = rays[i];
    rayOriginsAndMinT[i] = make_float4(r.getAnchor()[0],
                                       r.getAnchor()[1],
                                       r.getAnchor()[2],
                                       r.getInterval()[0]);

    float4 tempIn = make_float4(r.getDirection()[0],
                                r.getDirection()[1],
                                r.getDirection()[2],
                                r.getInterval()[1]);

    rayDirectionsAndMaxT[i] = tempIn;
  } // end for i

  // copy to the device stencil
  stdcuda::copy(stencil, stencil + n, deviceStencil.begin());

  stdcuda::vector_dev<float4> timeBarycentricsAndTriangleIndex(n);
  cudaRayTriangleBVHIntersection(NULL_NODE,
                                 mRootIndex,
                                 &rayOriginsAndMinT[0],
                                 &rayDirectionsAndMaxT[0],
                                 &mMinBoundHitIndexDevice[0],
                                 &mMaxBoundMissIndexDevice[0],
                                 &mFirstVertexDominantAxisDevice[0],
                                 &deviceStencil[0],
                                 &timeBarycentricsAndTriangleIndex[0],
                                 n);

  // copy deviceStencil to host
  stdcuda::copy(deviceStencil.begin(),
                deviceStencil.end(),
                stencil);

  // copy results to host
  std::vector<float4> results(n);
  stdcuda::copy(timeBarycentricsAndTriangleIndex.begin(),
                timeBarycentricsAndTriangleIndex.end(),
                &results[0]);

  // create Intersection objects
  for(size_t i = 0; i != n; ++i)
  {
    if(stencil[i])
    {
      float4 result = results[i];
      getIntersection(rays[i](result.x),
                      result.y, result.z,
                      floatAsUint(result.w),
                      intersections[i]);
    } // end if
  } // end for i
} // end CUDATriangleBVH::intersect()

