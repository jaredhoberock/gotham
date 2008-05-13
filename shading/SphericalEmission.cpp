/*! \file SphericalEmission.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of SphericalEmission class.
 */

#include "SphericalEmission.h"
#include "../geometry/Mappings.h"

#ifndef TWO_PI
#define TWO_PI 6.28318531f
#endif // TWO_PI

SphericalEmission
  ::SphericalEmission(const Spectrum &radiosity)
    :mRadiance(radiosity / TWO_PI)
{
  ;
} // end SphericalEmission::SphericalEmission()

Spectrum SphericalEmission
  ::evaluate(const Vector &w,
             const DifferentialGeometry &dg) const
{
  return mRadiance;
} // end SphericalEmission::evaluate()

Spectrum SphericalEmission
  ::sample(const DifferentialGeometry &dg,
           const float u0,
           const float u1,
           const float u2,
           Vector &w,
           float &pdf,
           bool &delta) const
{
  delta = false;
  Mappings<Vector>::unitSquareToSphere(u0, u1, dg.getTangent(), dg.getBinormal(), dg.getNormal(), w, pdf);
  return evaluate(w, dg);
} // end SphericalEmission::sample()

