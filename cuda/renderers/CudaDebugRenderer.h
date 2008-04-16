/*! \file CudaDebugRenderer.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a SIMD renderer which runs on
 *         a CUDA device.
 */

#pragma once

#include "../../renderers/SIMDRenderer.h"
#include "../../primitives/Intersection.h"
#include "../primitives/CudaIntersection.h"
#include <spectrum/Spectrum.h>
#include <stdcuda/stdcuda.h>

class ScatteringDistributionFunction;
class Ray;

class CudaDebugRenderer
  : public SIMDRenderer
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef SIMDRenderer Parent;

    /*! Null constructor calls the Parent
     */
    inline CudaDebugRenderer(void);

    /*! Constructor accepts a pointer to a Scene and Record.
     *  \param s Sets mScene.
     *  \param r Sets mRecord.
     */
    inline CudaDebugRenderer(boost::shared_ptr<const Scene>  s,
                             boost::shared_ptr<Record> r);

    /*! This method sets mScene.
     *  \param s Sets mScene.
     *  \note s must be a CudaScene.
     */
    inline virtual void setScene(const boost::shared_ptr<const Scene> &s);

    /*! This method sets mShadingContext.
     *  \param s Sets mShadingContext.
     *  \note s must be a CudaShadingContext.
     */
    inline virtual void setShadingContext(const boost::shared_ptr<ShadingContext> &s);

  protected:
    virtual void kernel(ProgressCallback &progress);

    virtual void sampleEyeRay(const size_t batchIdx,
                              const size_t threadIdx,
                              Ray *rays,
                              float *pdfs) const;

    virtual void shade(const Ray *rays,
                       const float *pdfs,
                       stdcuda::device_ptr<const CudaIntersection> intersectionsDevice,
                       stdcuda::device_ptr<const int> stencilDevice,
                       Spectrum *results,
                       const size_t n) const;

    virtual void deposit(const size_t batchIdx,
                         const size_t threadIdx,
                         const Spectrum *results);

    virtual void intersect(stdcuda::device_ptr<const float4> originsAndMinT,
                           stdcuda::device_ptr<const float4> directionsAndMaxT,
                           stdcuda::device_ptr<CudaIntersection> intersections,
                           stdcuda::device_ptr<int> stencil,
                           const size_t n);

}; // end CudaDebugRenderer

#include "CudaDebugRenderer.inl"

