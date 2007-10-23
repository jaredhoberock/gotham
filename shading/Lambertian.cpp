/*! \file Lambertian.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of Lambertian class.
 */

#include "Lambertian.h"
#include "../geometry/Mappings.h"

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

Spectrum Lambertian
  ::evaluate(const Vector &wo,
             const DifferentialGeometry &dg,
             const Vector &wi,
             const bool delta,
             const ComponentIndex component,
             float &pdf) const
{
  Spectrum result = Spectrum::black();
  pdf = 0;
  if(areSameHemisphere(wi, dg.getNormal(), wo))
  {
    pdf = Mappings::evaluateCosineHemispherePdf(wi, dg.getNormal());
    result = mAlbedoOverPi;
  } // end if

  return result;
} // end Lambertian::evaluate()

