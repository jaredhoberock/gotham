/*! \file ScatteredAccessContext.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a CudaShadingContext
 *         which accesses shaders in a scattered, incoherent manner.
 */

#pragma once

#include "CudaShadingContext.h"
#include <stdcuda/device_types.h>

class ScatteredAccessContext
  : public CudaShadingContext
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef CudaShadingContext Parent;

    /*! This method evaluates a batch of scattering shader jobs in SIMD fashion
     *  on a CUDA device.
     *  \param m A list of MaterialHandles.
     *  \param dg A list of CudaDifferentialGeometry objects.
     *  \param dgStride Stride size for elements in the dg list.
     *  \param stencil A stencil to control job processing.
     *  \param f The results of shading operations will be returned to this list.
     *  \param n The size of each list.
     */
    using Parent::evaluateScattering;
    virtual void evaluateScattering(const stdcuda::device_ptr<const MaterialHandle> &m,
                                    const stdcuda::device_ptr<const CudaDifferentialGeometry> &dg,
                                    const size_t dgStride,
                                    const stdcuda::device_ptr<const int> &stencil,
                                    const stdcuda::device_ptr<CudaScatteringDistributionFunction> &f,
                                    const size_t n);

    /*! This method evaluates a batch of emission shader jobs in SIMD fashion
     *  on a CUDA device.
     *  \param m A list of MaterialHandles.
     *  \param dg A list of CudaDifferentialGeometry objects.
     *  \param dgStride Stride size for elements in the dg list.
     *  \param stencil A stencil to control job processing.
     *  \param f The results of shading operations will be returned to this list.
     *  \param n The size of each list.
     */
    using Parent::evaluateEmission;
    virtual void evaluateEmission(const stdcuda::device_ptr<const MaterialHandle> &m,
                                  const stdcuda::device_ptr<const CudaDifferentialGeometry> &dg,
                                  const size_t dgStride,
                                  const stdcuda::device_ptr<const int> &stencil,
                                  const stdcuda::device_ptr<CudaScatteringDistributionFunction> &f,
                                  const size_t n);

    /*! This method evaluates a batch of sensor shader jobs in SIMD fashion
     *  on a CUDA device.
     *  \param m A list of MaterialHandles.
     *  \param dg A list of CudaDifferentialGeometry objects.
     *  \param dgStride Stride size for elements in the dg list.
     *  \param f The results of shading operations will be returned to this list.
     *  \param n The size of each list.
     */
    using Parent::evaluateSensor;
    virtual void evaluateSensor(const stdcuda::device_ptr<const MaterialHandle> &m,
                                const stdcuda::device_ptr<const CudaDifferentialGeometry> &dg,
                                const size_t dgStride,
                                const stdcuda::device_ptr<CudaScatteringDistributionFunction> &f,
                                const size_t n);
}; // end ScatteredAccessContext

