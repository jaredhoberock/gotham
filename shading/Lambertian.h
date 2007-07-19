/*! \file Lambertian.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a Lambertian
 *         scattering function.
 */

#ifndef LAMBERTIAN_H
#define LAMBERTIAN_H

#include "ScatteringDistributionFunction.h"

class Lambertian
  : public ScatteringDistributionFunction
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef ScatteringDistributionFunction Parent;

    /*! Constructor accepts an albedo.
     *  \param albedo Sets mAlbedo.
     */
    Lambertian(const Spectrum &albedo);

    /*! This method computes Lambertian scattering.
     *  \param wo The scattering direction.
     *  \param dg The DifferentialGeometry at the Point to be shaded.
     *  \param wi The incoming direction.
     *  \return mAlbedo / PI if wi & wo are in the same hemisphere;
     *          0, otherwise.
     */
    Spectrum evaluate(const Vector3 &wo,
                      const DifferentialGeometry &dg,
                      const Vector3 &wi) const;

  protected:
    Spectrum mAlbedo;

    /*! mAlbedo / PI
     */
    Spectrum mAlbedoOverPi;
}; // end Lambertian

#endif // LAMBERTIAN_H

