/*! \file UnitSquareToCone.inl
 *  \author Jared Hoberock
 *  \brief Inline file for UnitSquareToCone.h.
 */

#include "UnitSquareToCone.h"

#ifndef PI
#define PI 3.14159265f
#endif // PI

#ifndef INV_2PI
#define INV_2PI (1.0f / (2.0f * PI))
#endif // INV_2PI

template<typename Real, typename Real3>
  void UnitSquareToCone
    ::evaluate(const Real &theta,
               const Real &u,
               const Real &v,
               Real3 &p,
               Real *pdf)
{
  p[2] = 1.0f - theta*INV_2PI*u;
  Real r = sqrt(std::max<Real>(0, 1.0f - p[2]*p[2]));
  Real phi = 2.0f * PI * v;
  p[0] = r * cos(phi);
  p[1] = r * sin(phi);

  if(pdf != 0)
  {
    *pdf = evaluatePdf(p,theta);
  } // end if
} // end UnitSquareToCone::evaluate()

template<typename Real3, typename Real>
  Real UnitSquareToCone
    ::evaluatePdf(const Real3 &p,
                  const Real &theta)
{
  // the minimal value for a legit p[2] is:
  // 1.0f - theta*INV_2PI*1.0f
  if(p[2] <= 1.0f - theta*INV_2PI) return 0;

  // return one over the area of the cone of directions
  return 1.0f / theta;
} // end UnitSquareToCone::evaluatePdf()

