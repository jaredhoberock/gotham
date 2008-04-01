/*! \file HemisphericalEmission.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to an EmissionFunction
 *         which emits a constant radiance uniformly over
 *         a hemisphere.
 */

#ifndef HEMISPHERICAL_EMISSION_H
#define HEMISPHERICAL_EMISSION_H

#include "ScatteringDistributionFunction.h"

class HemisphericalEmission
  : public ScatteringDistributionFunction
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef ScatteringDistributionFunction Parent;

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
    using Parent::evaluate;
    Spectrum evaluate(const Vector3 &w,
                      const DifferentialGeometry &dg) const;

    /*! This method returns a const reference to mRadiance.
     *  \return mRadiance
     */
    const Spectrum &getRadiance(void) const;
    
  protected:
    Spectrum mRadiance;
}; // end HemisphericalEmission

#endif // HEMISPHERICAL_EMISSION_H

