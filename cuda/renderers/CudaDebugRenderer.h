/*! \file CudaDebugRenderer.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a SIMD renderer which runs on
 *         a CUDA device.
 */

#pragma once

#include "CudaRenderer.h"
#include "../primitives/CudaIntersection.h"
#include <spectrum/Spectrum.h>
#include <stdcuda/stdcuda.h>
#include <vector>

class CudaDebugRenderer
  : public CudaRenderer
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef CudaRenderer Parent;

    /*! Null constructor calls the Parent
     */
    inline CudaDebugRenderer(void);

    /*! Constructor accepts a pointer to a Scene and Record.
     *  \param s Sets mScene.
     *  \param r Sets mRecord.
     */
    inline CudaDebugRenderer(boost::shared_ptr<const Scene>  s,
                             boost::shared_ptr<Record> r);

  protected:
    virtual void kernel(ProgressCallback &progress);

    virtual void sampleEyeRays(const stdcuda::device_ptr<const float4> &u0,
                               const stdcuda::device_ptr<const float4> &u1,
                               const stdcuda::device_ptr<float4> &originsAndMinT,
                               const stdcuda::device_ptr<float4> &directionsAndMaxT,
                               const stdcuda::device_ptr<float> &pdfs,
                               const size_t n) const;

    virtual void generateHyperPoints(stdcuda::vector_dev<float4> &u0,
                                     stdcuda::vector_dev<float4> &u1,
                                     const size_t n) const;

    virtual void shade(stdcuda::device_ptr<const float4> directionsAndMaxT,
                       stdcuda::device_ptr<const CudaIntersection> intersectionsDevice,
                       stdcuda::device_ptr<const int> stencilDevice,
                       stdcuda::device_ptr<float3> results,
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

