/*! \file UnitSquareToHemisphere.inl
 *  \author Jared Hoberock
 *  \brief Inline file for UnitSquareToHemisphere.h.
 */

#include "UnitSquareToHemisphere.h"

#include <math.h>

#ifndef PI
#define PI 3.14159265f
#endif // PI

#ifndef INV_2PI
#define INV_2PI (1.0f / (2.0f * PI))
#endif // INV_PI

#ifndef DEGREES
#define DEGREES(X) (X * (180.0f / PI))
#endif // DEGREES

template<typename Real, typename Point3>
  void UnitSquareToHemisphere
    ::evaluate(const Real &u,
               const Real &v,
               Point3 &p,
               Real *pdf)
{
  // FIXME: kill these floats
  p[2] = 1.0f - u;
  Real r = sqrt(std::max<Real>(Real(0.000005), u*(2.0f - u)));
  Real phi = 2.0f * PI * v;
  p[0] = r * cos(phi);
  p[1] = r * sin(phi);

  if(pdf != 0)
  {
    *pdf = INV_2PI;
  } // end if
} // end UnitSquareToHemisphere::evaluate()

template<typename Point3, typename Real>
  void UnitSquareToHemisphere
     ::inverse(const Point3 &p,
               Real &u,
               Real &v)
{
  u = 1.0f - p[2];
  float r = sqrt(std::max(0.000005f, 1.0f - u*u));

  float phi = atan2(p[1] / r, p[0] / r);
  if(phi < 0.0f) phi += 2.0f * PI;

  v = phi / (2.0f * PI);
} // end UnitSquareToHemisphere::inverse()

template<typename Real, typename Real3>
  Real UnitSquareToHemisphere
    ::evaluatePdf(const Real3 &p)
{
  return INV_2PI;
} // end UnitSquareToHemisphere::evaluatePdf()

