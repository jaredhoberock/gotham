/*! \file CUDATriangleBVH.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a TriangleBVH
 *         with facilities for SIMD processing on a
 *         CUDA-capable gpu.
 */

#ifndef CUDA_TRIANGLE_BVH_H
#define CUDA_TRIANGLE_BVH_H

#include "TriangleBVH.h"

// this defines the CUDA vector types
#include <vector_types.h>
#include <stdcuda/vector_dev.h>

class CUDATriangleBVH
  : public TriangleBVH
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef TriangleBVH Parent;

    /*! This method provides a SIMD path for intersect(). It intersects more than
     *  one Ray against this Primitive en masse.
     *  \param rays A list of Rays to intersect.
     *  \param intersections Af an intersection for a Ray exists, a Primitive::Intersection record storing information about the first
     *         intersection encountered is returned here.
     *  \param stencil If a Ray hits something, this is set to true.
     *  \param n The length of lists rays, intersections, and stencil.
     *           n must be less than or equal to mWorkBatchSize. Any additional
     *           work will be ignored.
     */
    virtual void intersect(Ray *rays,
                           Intersection *intersections,
                           int *stencil,
                           size_t n) const;


    /*! This method intializes various CUDA data structures to prepare
     *  for processing.
     */
    virtual void finalize(void);

    /*! This method sets mWorkBatchSize.
     *  \param b Sets mWorkBatchSize.
     */
    void setWorkBatchSize(const size_t b);

  protected:
    /*! This method creates the per-triangle data which do is
     *  not involved with ray-triangle intersection.
     */
    virtual void createPerTriangleGeometricData();

    /*! This method creates the CUDA scratch space needed
     *  for each call to intersect().
     */
    virtual void createScratchSpace(void);

    /*! This method transforms the results of cudaRayTriangleBVHIntersection()
     *  into Intersection objects.
     *  \param rayOriginsAndMinT A list of ray origins.
     *  \param rayDirectionsAndMaxT A list of ray directions.
     *  \param timeBarycentricsAndTriangleIndex The results of cudaRayTriangleBVHIntersection.
     *  \param stencil A filter to control which results need to be written.
     *  \param intersections If an intersection for a Ray exists, a Primitive::Intersection record storing information about the first
     *         intersection encountered is returned here.
     *  \param n The length of the lists.
     */
    virtual void createIntersections(stdcuda::const_device_pointer<float4> rayOriginsAndMinT,
                                     stdcuda::const_device_pointer<float4> rayDirectionsAndMaxT,
                                     stdcuda::const_device_pointer<float4> timeBarycentricsAndTriangleIndex,
                                     stdcuda::const_device_pointer<int> stencil,
                                     Intersection *intersections,
                                     const size_t n) const;

    // These are copies of the corresponding lists in the Parents which
    // are resident on the CUDA device
    stdcuda::vector_dev< ::float4> mMinBoundHitIndexDevice;
    stdcuda::vector_dev< ::float4> mMaxBoundMissIndexDevice;
    stdcuda::vector_dev< ::float4> mFirstVertexDominantAxisDevice;

    // these are per-triangle lists of data
    // XXX some of this is redundant and should be per-primitive
    stdcuda::vector_dev< ::float3>       mGeometricNormalDevice;
    stdcuda::vector_dev< ::float2>       mFirstVertexParmsDevice;
    stdcuda::vector_dev< ::float2>       mSecondVertexParmsDevice;
    stdcuda::vector_dev< ::float2>       mThirdVertexParmsDevice;
    stdcuda::vector_dev<float>           mPrimitiveInvSurfaceAreaDevice;
    stdcuda::vector_dev<PrimitiveHandle> mPrimitiveHandlesDevice;

    /*! This parameter controls the batch size of the workload.
     */
    size_t mWorkBatchSize;

    /*! Scratch space for intersect().
     *  Make these mutable because their contents will be modified
     *  in intersect().
     */
    mutable stdcuda::vector_dev<float4> mRayOriginsAndMinT;
    mutable stdcuda::vector_dev<float4> mRayDirectionsAndMaxT;
    mutable stdcuda::vector_dev<int>    mDeviceStencil;
    mutable stdcuda::vector_dev<float4> mTimeBarycentricsAndTriangleIndex;
}; // end CUDATriangleBVH

#endif // CUDA_TRIANGLE_BVH_H

