/*! \file Lambertian.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a Lambertian
 *         scattering function.
 */

#ifndef LAMBERTIAN_H
#define LAMBERTIAN_H

#include "ScatteringFunction.h"

class Lambertian
  : public ScatteringFunction
{
  public:
    /*! Constructor accepts an albedo.
     *  \param albedo Sets mAlbedo.
     */
    Lambertian(const Spectrum &albedo);

    /*! This method computes Lambertian scattering.
     *  \param wi The incoming direction.
     *  \param wo The scattering direction.
     *  \return mAlbedo / PI if wi & wo are in the same hemisphere;
     *          0, otherwise.
     */
    Spectrum evaluate(const Vector3 &wi,
                      const Vector3 &wo) const;

  protected:
    Spectrum mAlbedo;

    /*! mAlbedo / PI
     */
    Spectrum mAlbedoOverPi;
}; // end Lambertian

#endif // LAMBERTIAN_H

