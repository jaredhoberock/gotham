/*! \file UnitSquareToSphere.inl
 *  \author Jared Hoberock
 *  \brief Inline file for UnitSquareToSphere.h.
 */

#include <math.h>
#include "UnitSquareToSphere.h"

#ifndef PI
#define PI 3.14159265f
#endif // PI

template<typename Real, typename Real3>
  void UnitSquareToSphere
    ::evaluate(const Real &u,
               const Real &v,
               Real3 &p,
               Real *pdf)
{
  p[2] = static_cast<Real>(1.0) - static_cast<Real>(2.0) * u;
  Real r = Real(1) - p[2]*p[2];
  r = (r > 0) ? r : 0;
  r = sqrt(r);
  Real phi = static_cast<Real>(2.0) * PI * v;
  p[0] = r * cos(phi);
  p[1] = r * sin(phi);

  if(pdf != 0)
  {
    *pdf = evaluatePdf<Real,Real3>(p);
  } // end if
} // end UnitSquareToSphere::evaluate()

template<typename Real, typename Real3>
  Real UnitSquareToSphere
    ::evaluatePdf(const Real3 &p)
{
  return static_cast<float>(1.0) / (static_cast<float>(4.0) * PI);
} // end UnitSquareToSphere::evaluatePdf()

