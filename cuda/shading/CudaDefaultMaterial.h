/*! \file CudaDefaultMaterial.h
 *  \author Jared Hoberock
 *  \brief This class serves as a default CudaMaterial
 *         with Lambertian scattering that we are guaranteed
 *         always exists.
 */

#pragma once

#include "../include/CudaMaterial.h"

class CudaDefaultMaterial
  : public CudaMaterial
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef CudaMaterial Parent;

    virtual const char *getName(void) const;

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
                                    const CudaDifferentialGeometryArray &dg,
                                    const stdcuda::device_ptr<const bool> &stencil,
                                    const stdcuda::device_ptr<CudaScatteringDistributionFunction> &f,
                                    const size_t n) const;
}; // end CudaDefaultMaterial

