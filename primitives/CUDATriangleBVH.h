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
     *  \param rays A list of Rays to intersect.
     *  \param intersections Af an intersection for a Ray exists, a Primitive::Intersection record storing information about the first
     *         intersection encountered is returned here.
     *  \param stencil If a Ray hits something, this is set to true.
     *  \param n The length of lists rays, intersections, and stencil.
     *
     *  XXX this will require a malloc each time since we don't know n
     *      in advance.  consider an alternative where n is fixed
     *      we could fetch the number of threads in the api and init that way
     */
    virtual void intersect(Ray *rays,
                           Intersection *intersections,
                           int *stencil,
                           const size_t n) const;


    /*! This method intializes various CUDA data structures to prepare
     *  for processing.
     */
    virtual void finalize(void);

  protected:
    // These are copies of the corresponding lists in the Parents which
    // are resident on the CUDA device
    stdcuda::vector_dev< ::float4> mMinBoundHitIndexDevice;
    stdcuda::vector_dev< ::float4> mMaxBoundMissIndexDevice;
    stdcuda::vector_dev< ::float4> mFirstVertexDominantAxisDevice;
}; // end CUDATriangleBVH

#endif // CUDA_TRIANGLE_BVH_H

