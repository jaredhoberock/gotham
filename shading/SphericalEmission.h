/*! \file SphericalEmission.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to an EmissionFunction
 *         which emits a constant radiance uniformly over
 *         a sphere.
 */

#ifndef SPHERICAL_EMISSION_H
#define SPHERICAL_EMISSION_H

#include "ScatteringDistributionFunction.h"

class SphericalEmission
  : public ScatteringDistributionFunction
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef ScatteringDistributionFunction Parent;

    /*! Constructor accepts a radiance.
     *  \param radiosity The sum of radiance emitted from this
     *         hemisphere.  mRadiance is set to radiosity / (2PI)
     */
    SphericalEmission(const Spectrum &radiosity);

    /*! This method samples the emission at dg.
     *  \param dg The DifferentialGeometry at the point of interest.
     *  \param u0 A real number in [0,1).
     *  \param u1 A second real number in [0,1).
     *  \param u2 A third real number in [0,1).
     *  \param w The direction of scattering is returned here.
     *  \param pdf The value of the pdf at (u0,u1,u2) is returned here.
     *  \param delta This is set to false.
     *  \return The unidirectional scattering to w is returned here.
     */
    virtual Spectrum sample(const DifferentialGeometry &dg,
                            const float u0,
                            const float u1,
                            const float u2,
                            Vector3 &w,
                            float &pdf,
                            bool &delta) const;

    /*! This function returns mRadiance.
     *  \param w  The outgoing direction of emission.
     *  \param dg The DifferentialGeometry at the Point of interest.
     */
    using Parent::evaluate;
    Spectrum evaluate(const Vector3 &w,
                      const DifferentialGeometry &dg) const;
    
  protected:
    Spectrum mRadiance;
}; // end SphericalEmission

#endif // SPHERICAL_EMISSION_H

