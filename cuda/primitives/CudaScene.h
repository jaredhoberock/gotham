/*! \file CudaScene.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a Scene
 *         with a primitive residing on a CUDA
 *         device.
 */

#pragma once

#include "../../primitives/Scene.h"
#include "CudaIntersection.h"
#include <stdcuda/stdcuda.h>

class CudaScene
  : public Scene
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef Scene Parent;

    /*! This method provides a SIMD path for ray intersection
     *  for ray data that resides on a CUDA device.
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

    /*! This method sets mPrimitive.
     *  \param g Sets mPrimitive.
     */
    virtual void setPrimitive(boost::shared_ptr<Primitive> g);
}; // end CudaScene

