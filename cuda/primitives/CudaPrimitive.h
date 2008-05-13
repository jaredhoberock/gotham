/*! \file CudaPrimitive.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a Primitive
 *         residing on a CUDA device.
 */

#pragma once

#include <stdcuda/stdcuda.h>
#include "CudaIntersection.h"
#include "../geometry/CudaDifferentialGeometryArray.h"

class CudaPrimitive
{
  public:
    /*! Null destructor does nothing.
     */
    inline virtual ~CudaPrimitive(void){};

    /*! This method provides a SIMD path for intersect() with
     *  types that explicitly reside on a CUDA device. It intersects
     *  a set of Rays against this CudaPrimitive en masse.
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
                           const size_t n) const = 0;

    /*! This method provides a SIMD path for intersect() with
     *  types that explicitly reside on a CUDA device. It intersects
     *  a set of Rays against this CudaPrimitive en masse.
     *  \param directions A list of ray directions.
     *  \param interval An interval over which a ray intersection is valid.
     *         This interval is shared by all rays.
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
                           const size_t n) const = 0;


    /*! This method provides a SIMD path for shadow ray intersection
     *  for ray data that resides on a CUDA device.
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
                           const size_t n) const = 0;

    /*! This method provides a SIMD path for shadow ray intersection
     *  for ray data that resides on a CUDA device.
     *  \param origins A list of ray origins.
     *  \param directions A list of ray directions.
     *  \param interval An interval over which a ray intersection is valid.
     *         This interval is shared by all rays.
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
                           const size_t n) const = 0;
}; // end CudaPrimitive
