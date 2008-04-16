/*! \file CUDATriangleBVH.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a TriangleBVH
 *         with facilities for SIMD processing on a
 *         CUDA-capable gpu.
 */

#ifndef CUDA_TRIANGLE_BVH_H
#define CUDA_TRIANGLE_BVH_H

#include "../../primitives/TriangleBVH.h"
#include "CudaPrimitive.h"
#include "CudaTriangleList.h"

// this defines the CUDA vector types
#include <vector_types.h>
#include <stdcuda/vector_dev.h>

class CUDATriangleBVH
  : public TriangleBVH,
    public CudaPrimitive,
    public CudaTriangleList
{
  public:
    /*! \typedef Parent0
     *  \brief Shorthand.
     */
    typedef TriangleBVH Parent0;

    /*! \typedef Parent1
     *  \brief Shorthand.
     */
    typedef CudaPrimitive Parent1;

    /*! \typedef Parent2
     *  \brief Shorthand.
     */
    typedef CudaTriangleList Parent2;

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

    /*! This method provides a SIMD path for intersect() with
     *  types that explicitly reside on a CUDA device. It intersects
     *  a set of Rays against this CudaPrimitive en masse.
     *  \param originsAndMinT A list of ray origins. The fourth
     *         component is interpreted as the minimum of the
     *         valid parametric interval.
     *  \param directionsAndMaxT A list of ray directions. The fourth
     *         component is interpreted as the maximum of the
     *         valid parametric interval.
     *  \param intersections If an intersection for a Ray exists,
     *         a CudaIntersection record storing information about
     *         the first intersection encountered is returned here.
     *  \param stencil If a Ray hits something, this is set to true.
     *  \param n The length of the lists.
     */
    virtual void intersect(stdcuda::device_ptr<const float4> originsAndMinT,
                           stdcuda::device_ptr<const float4> directionsAndMaxT,
                           stdcuda::device_ptr<CudaIntersection> intersections,
                           stdcuda::device_ptr<int> stencil,
                           const size_t n) const;

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

    /*! This method creates the list which maps PrimitiveHandles to
     *  MaterialHandles.
     */
    virtual void createPrimitiveHandleToMaterialHandleMap();

    // These are copies of the corresponding lists in the Parents which
    // are resident on the CUDA device
    stdcuda::vector_dev< ::float4> mMinBoundHitIndexDevice;
    stdcuda::vector_dev< ::float4> mMaxBoundMissIndexDevice;
    stdcuda::vector_dev< ::float4> mFirstVertexDominantAxisDevice;

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

