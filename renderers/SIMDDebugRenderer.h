/*! \file SIMDDebugRenderer.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a SIMD renderer which implements the
 *         same functionality as DebugRenderer.
 */

#ifndef SIMD_DEBUG_RENDERER_H
#define SIMD_DEBUG_RENDERER_H

#include "SIMDRenderer.h"
#include "../primitives/Primitive.h"
#include "../primitives/Primitive.h"
#include "../include/Spectrum.h"

class ScatteringDistributionFunction;

class SIMDDebugRenderer
  : public SIMDRenderer
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef SIMDRenderer Parent;

    /*! Null constructor calls the Parent
     */
    inline SIMDDebugRenderer(void);

    /*! Constructor accepts a pointer to a Scene and Record.
     *  \param s Sets mScene.
     *  \param r Sets mRecord.
     */
    inline SIMDDebugRenderer(boost::shared_ptr<const Scene>  s,
                             boost::shared_ptr<Record> r);

  protected:
    virtual void kernel(ProgressCallback &progress);

    virtual void sampleEyeRay(const size_t batchIdx,
                              const size_t threadIdx,
                              Ray *rays,
                              float *pdfs) const;

    virtual void shade(const Ray *rays,
                       const float *pdfs,
                       const Intersection *intersections,
                       const bool *stencil,
                       Spectrum *results,
                       const size_t n) const;

    virtual void deposit(const size_t batchIdx,
                         const size_t threadIdx,
                         const Spectrum *results);

    virtual void intersect(Ray *rays,
                           Intersection *intersections,
                           bool *stencil);

}; // end SIMDDebugRenderer

#include "SIMDDebugRenderer.inl"

#endif // SIMD_DEBUG_RENDERER_H

