/*! \file CUDATriangleBVH.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of CUDATriangleBVH class.
 */

#include "CUDATriangleBVH.h"
#include <vector_functions.h>
#include <stdcuda/stdcuda.h>
#include "cudaRayTriangleBVHIntersection.h"
#include <stdcuda/cuda_algorithm.h>

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
  stdcuda::vector_dev<float4>    rayOriginsAndMinT(n);
  stdcuda::vector_dev<float4> rayDirectionsAndMaxT(n);
  stdcuda::vector_dev<int> deviceStencil(n);

  for(size_t i = 0; i != n; ++i)
  {
    const Ray &r = rays[i];
    rayOriginsAndMinT[i] = make_float4(r.getAnchor()[0],
                                       r.getAnchor()[1],
                                       r.getAnchor()[2],
                                       r.getInterval()[0]);

    rayDirectionsAndMaxT[i] = make_float4(r.getDirection()[0],
                                          r.getDirection()[1],
                                          r.getDirection()[2],
                                          r.getInterval()[1]);
  } // end for i
  //std::cerr << "init device rays" << std::endl;
  //std::cerr << "CUDA error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;

  // copy to the device stencil
  stdcuda::copy(stencil, stencil + n, deviceStencil.begin());
  //std::cerr << "copied to device stencil" << std::endl;
  //std::cerr << "CUDA error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;

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
  //std::cerr << "back from kernel" << std::endl;
  //std::cerr << "CUDA error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;

  // copy deviceStencil to host
  stdcuda::copy(deviceStencil.begin(),
                deviceStencil.end(),
                stencil);
  //std::cerr << "copied stencil to host" << std::endl;
  //std::cerr << "CUDA error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;

  // copy results to host
  std::vector<float4> results(n);
  stdcuda::copy(timeBarycentricsAndTriangleIndex.begin(),
                timeBarycentricsAndTriangleIndex.end(),
                &results[0]);
  //std::cerr << "copied results to host" << std::endl;
  //std::cerr << "CUDA error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;

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

  //std::cerr << "CUDATriangleBVH::intersect(): exiting" << std::endl;
  //std::cerr << "CUDA error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
} // end CUDATriangleBVH::intersect()

