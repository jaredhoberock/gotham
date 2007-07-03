/*! \file Lambertian.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of Lambertian class.
 */

#include "Lambertian.h"

Lambertian
  ::Lambertian(const Spectrum &albedo)
{
  mAlbedo = albedo;
  mAlbedoOverPi = mAlbedo / PI;
} // end Lambertian::Lambertian()

Spectrum Lambertian
  ::evaluate(const Vector3 &wi,
             const Vector3 &wo) const
{
  return areSameHemisphere(wi,wo) ? mAlbedoOverPi : Spectrum::black();
} // end Lambertian::evaluate()

