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
                           stdcuda::device_ptr<int> stencil,
                           const size_t n) const = 0;
}; // end CudaPrimitive

