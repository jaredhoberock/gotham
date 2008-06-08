/*! \file HemisphericalEmission.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to an EmissionFunction
 *         which emits a constant radiance uniformly over
 *         a hemisphere.
 */

#ifndef HEMISPHERICAL_EMISSION_H
#define HEMISPHERICAL_EMISSION_H

#include "ScatteringDistributionFunction.h"
#include "functions/HemisphericalEmissionBase.h"

class HemisphericalEmission
  : public ScatteringDistributionFunction,
    public HemisphericalEmissionBase<Vector,Spectrum>
{
  public:
    /*! \typedef Parent0
     *  \brief Shorthand.
     */
    typedef ScatteringDistributionFunction Parent0;

    /*! \typedef Parent1
     *  \brief Shorthand.
     */
    typedef HemisphericalEmissionBase<Vector,Spectrum> Parent1;

    /*! Constructor accepts a radiance.
     *  \param radiosity The sum of radiance emitted from this
     *         hemisphere.  mRadiance is set to radiosity / PI
     */
    HemisphericalEmission(const Spectrum &radiosity);

    /*! This function returns mRadiance when we
     *  is in the positive hemisphere; Spectrum::black(),
     *  otherwise.
     *  \param w  The outgoing direction of emission.
     *  \param dg The DifferentialGeometry at the Point of interest.
     */
    using Parent0::evaluate;
    Spectrum evaluate(const Vector &w,
                      const DifferentialGeometry &dg) const;
}; // end HemisphericalEmission

#endif // HEMISPHERICAL_EMISSION_H

