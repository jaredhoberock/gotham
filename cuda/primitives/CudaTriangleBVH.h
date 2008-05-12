/*! \file CudaTriangleBVH.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a TriangleBVH
 *         with facilities for SIMD processing on a
 *         Cuda-capable gpu.
 */

#pragma once

#include "../../primitives/TriangleBVH.h"
#include "CudaPrimitive.h"
#include "CudaTriangleList.h"

// this defines the Cuda vector types
#include <vector_types.h>
#include <stdcuda/vector_dev.h>

class CudaTriangleBVH
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

    /*! This method provides a SIMD path for intersect() with
     *  types that explicitly reside on a Cuda device. It intersects
     *  a set of rays against this CudaTriangleBVH en masse.
     *  \param origins A list of ray origins.
     *  \param directions A list of ray directions.
     *  \param intervals A list of intervals over which a ray intersection is valid.
     *  \param dg If an intersection exists, the differential geometry at the
     *         section point is returned to these lists.
     *  \param hitPrims If an intersection exists, the hit primitive is returned to
     *         this list.
     *  \param stencil If a Ray hits something, this is set to true.
     *  \param n The length of the lists.
     */
    virtual void intersect(const stdcuda::device_ptr<const float3> &origins,
                           const stdcuda::device_ptr<const float3> &directions,
                           const stdcuda::device_ptr<const float2> &intervals,
                           CudaDifferentialGeometryArray &dg,
                           const stdcuda::device_ptr<PrimitiveHandle> &hitPrims,
                           const stdcuda::device_ptr<bool> &stencil,
                           const size_t n) const;

    /*! This method provides a SIMD path for intersect() with
     *  types that explicitly reside on a Cuda device. It intersects
     *  a set of rays against this CudaTriangleBVH en masse.
     *  \param origins A list of ray origins.
     *  \param directions A list of ray directions.
     *  \param interval An intervals over which a ray intersection is valid.
     *         This is shared by all rays.
     *  \param dg If an intersection exists, the differential geometry at the
     *         section point is returned to these lists.
     *  \param hitPrims If an intersection exists, the hit primitive is returned to
     *         this list.
     *  \param stencil If a Ray hits something, this is set to true.
     *  \param n The length of the lists.
     */
    virtual void intersect(const stdcuda::device_ptr<const float3> &origins,
                           const stdcuda::device_ptr<const float3> &directions,
                           const float2 &interval,
                           CudaDifferentialGeometryArray &dg,
                           const stdcuda::device_ptr<PrimitiveHandle> &hitPrims,
                           const stdcuda::device_ptr<bool> &stencil,
                           const size_t n) const;

    /*! This method provides a SIMD path for shadow ray intersection
     *  for ray data that resides on a Cuda device.
     *  \param origins A list of ray origins.
     *  \param directions A list of ray directions.
     *  \param intervals A list of intervals over which a ray intersection is valid.
     *  \param stencil This mask controls processing.
     *  \param results If stencil is set to 0, this is set to 0. Otherwise,
     *                 it is set to 0 if the ray hits something; 1, otherwise.
     *  \param n The length of the lists.
     */
    virtual void intersect(const stdcuda::device_ptr<const float3> &origins,
                           const stdcuda::device_ptr<const float3> &directions,
                           const stdcuda::device_ptr<const float2> &intervals,
                           const stdcuda::device_ptr<const bool> &stencil,
                           const stdcuda::device_ptr<bool> &results,
                           const size_t n) const;

    /*! This method provides a SIMD path for shadow ray intersection
     *  for ray data that resides on a Cuda device.
     *  \param origins A list of ray origins.
     *  \param directions A list of ray directions.
     *  \param interval An interval over which a ray intersection is valid.
     *         This is shared by all rays.
     *  \param stencil This mask controls processing.
     *  \param results If stencil is set to 0, this is set to 0. Otherwise,
     *                 it is set to 0 if the ray hits something; 1, otherwise.
     *  \param n The length of the lists.
     */
    virtual void intersect(const stdcuda::device_ptr<const float3> &origins,
                           const stdcuda::device_ptr<const float3> &directions,
                           const float2 &interval,
                           const stdcuda::device_ptr<const bool> &stencil,
                           const stdcuda::device_ptr<bool> &results,
                           const size_t n) const;


    /*! This method intializes various Cuda data structures to prepare
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

    /*! This method creates the Cuda scratch space needed
     *  for each call to intersect().
     */
    virtual void createScratchSpace(void);

    /*! This method creates the list which maps PrimitiveHandles to
     *  MaterialHandles.
     */
    virtual void createPrimitiveHandleToMaterialHandleMap();

    // These are copies of the corresponding lists in the Parents which
    // are resident on the Cuda device
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
    mutable stdcuda::vector_dev<float4> mTimeBarycentricsAndTriangleIndex;
}; // end CudaTriangleBVH

