/*! \file CudaMaterial.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a Material class
 *         with an interface for doing SIMD shading on
 *         a CUDA device.
 */

#pragma once

#include "../../include/Material.h"
#include <stdcuda/device_types.h>
#include "../geometry/CudaDifferentialGeometry.h"
#include "../shading/CudaScatteringDistributionFunction.h"
class CudaShadingInterface;

class CudaMaterial
  : public Material
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef Material Parent;

    /*! This method evaluates this CudaMaterial's scattering shader on a
     *  list of CudaDifferentialGeometrys in SIMD fashion.
     *  \param context A context for shader evaluation.
     *  \param dg A list of shading points.
     *  \param stencil A stencil to control which jobs get processed.
     *  \param f The result of each shading job is returned to this list.
     *  \param n The size of the lists.
     */
    using Parent::evaluateScattering;
    virtual void evaluateScattering(CudaShadingInterface &context,
                                    const stdcuda::device_ptr<const CudaDifferentialGeometry> &dg,
                                    const stdcuda::device_ptr<const int> &stencil,
                                    const stdcuda::device_ptr<CudaScatteringDistributionFunction> &f,
                                    const size_t n) const;

    virtual void evaluateScattering(CudaShadingInterface &context,
                                    const stdcuda::device_ptr<const CudaDifferentialGeometry> &dg,
                                    const size_t dgStride,
                                    const stdcuda::device_ptr<const int> &stencil,
                                    const stdcuda::device_ptr<CudaScatteringDistributionFunction> &f,
                                    const size_t n) const;

    /*! This method evaluates this CudaMaterial's emission shader on a
     *  list of CudaDifferentialGeometrys in SIMD fashion.
     *  \param context A context for shader evaluation.
     *  \param dg A list of shading points.
     *  \param dgStride The stride size between elements in the dg list.
     *  \param stencil A stencil to control which jobs get processed.
     *  \param f The result of each shading job is returned to this list.
     *  \param n The size of the lists.
     */
    using Parent::evaluateEmission;
    virtual void evaluateEmission(CudaShadingInterface &context,
                                  const stdcuda::device_ptr<const CudaDifferentialGeometry> &dg,
                                  const size_t dgStride,
                                  const stdcuda::device_ptr<const int> &stencil,
                                  const stdcuda::device_ptr<CudaScatteringDistributionFunction> &f,
                                  const size_t n) const;

    /*! This method evaluates this CudaMaterial's sensor shader on a
     *  list of CudaDifferentialGeometrys in SIMD fashion.
     *  \param context A context for shader evaluation.
     *  \param dg A list of shading points.
     *  \param stencil A stencil to control which jobs get processed.
     *  \param f The result of each shading jo is returned to this list.
     *  \param n The size of the lists.
     */
    using Parent::evaluateSensor;
    virtual void evaluateSensor(CudaShadingInterface &context,
                                const stdcuda::device_ptr<const CudaDifferentialGeometry> &dg,
                                const size_t dgStride,
                                const stdcuda::device_ptr<const int> &stencil,
                                const stdcuda::device_ptr<CudaScatteringDistributionFunction> &f,
                                const size_t n) const;
}; // end CudaMaterial

