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
  p[2] = Real(1) - u;
  Real r = u*(Real(2) - u);
  r = (r > Real(0.000005)) ? r : Real(0.000005);
  Real phi = Real(2) * PI * v;
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
  u = Real(1) - p[2];
  float r = Real(1) - u*u;
  r = (r > Real(0.000005)) ? r : Real(0.000005);

  float phi = atan2(p[1] / r, p[0] / r);
  if(phi < 0) phi += Real(2) * PI;

  v = phi / (Real(2) * PI);
} // end UnitSquareToHemisphere::inverse()

template<typename Real, typename Real3>
  Real UnitSquareToHemisphere
    ::evaluatePdf(const Real3 &p)
{
  return INV_2PI;
} // end UnitSquareToHemisphere::evaluatePdf()

