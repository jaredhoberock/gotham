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

    using Parent::evaluateScattering;
    virtual void evaluateScattering(CudaShadingInterface &context,
                                    const stdcuda::device_ptr<const CudaDifferentialGeometry> &dg,
                                    const size_t dgStride,
                                    const stdcuda::device_ptr<const int> &stencil,
                                    const stdcuda::device_ptr<CudaScatteringDistributionFunction> &f,
                                    const size_t n) const;
}; // end CudaDefaultMaterial

