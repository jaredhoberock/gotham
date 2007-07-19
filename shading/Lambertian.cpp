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
  ::evaluate(const Vector3 &wo,
             const DifferentialGeometry &dg,
             const Vector3 &wi) const
{
  return areSameHemisphere(wi,dg.getNormal(),wo) ? mAlbedoOverPi : Spectrum::black();
} // end Lambertian::evaluate()

