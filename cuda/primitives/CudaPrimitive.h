/*! \file CudaPrimitive.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a Primitive
 *         residing on a CUDA device.
 */

#pragma once

#include <stdcuda/stdcuda.h>
#include "CudaIntersection.h"

class CudaPrimitive
{
  public:
    /*! Null destructor does nothing.
     */
    inline virtual ~CudaPrimitive(void){};

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
                           stdcuda::device_ptr<bool> stencil,
                           const size_t n) const = 0;

    /*! This method provides a SIMD path for shadow ray intersection
     *  for ray data that resides on a CUDA device.
     *  \param originsAndMinT A list of ray origins. The fourth
     *         component is interpreted as the minimum of the
     *         valid parametric interval.
     *  \param directionsAndMaxT A list of ray directions. The fourth
     *         component is interpreted as the maximum of the valid
     *         parametric interval.
     *  \param stencil This mask controls processing.
     *  \param results If stencil is set to 0, this is set to 0. Otherwise,
     *                 it is set to 0 if the ray hits something; 1, otherwise.
     *  \param n The length of the lists.
     */
    virtual void intersect(const stdcuda::device_ptr<const float4> &originsAndMinT,
                           const stdcuda::device_ptr<const float4> &directionsAndMaxT,
                           const stdcuda::device_ptr<const bool> &stencil,
                           const stdcuda::device_ptr<bool> &results,
                           const size_t n) const = 0;
}; // end CudaPrimitive

