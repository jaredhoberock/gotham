/*! \file UnitSquareToIsoscelesRightTriangle.inl
 *  \author Jared Hoberock
 *  \brief Inline file for UnitSquareToIsoscelesRightTriangle.h.
 */

#include "UnitSquareToIsoscelesRightTriangle.h"
#include <math.h>

template<typename Real, typename Real2>
  void UnitSquareToIsoscelesRightTriangle
    ::evaluate(const Real &u,
               const Real &v,
               Real2 &p,
               Real *pdf)
{
  Real su = sqrt(u);
  p[0] = Real(1) - su;
  p[1] = v * su;

  if(pdf)
  {
    *pdf = evaluatePdf<Real,Real2>(p);
  } // end if
} // end UnitSquareToIsoscelesRightTriangle::evaluate()

template<typename Real2, typename Real>
  void UnitSquareToIsoscelesRightTriangle
    ::inverse(const Real2 &p,
              Real &u,
              Real &v)
{
  Real su = Real(1) - p[0];
  u = su*su;
  v = p[1] / su;
} // end UnitSquareToIsoscelesRightTriangle::inverse()

template<typename Real, typename Real2>
  Real UnitSquareToIsoscelesRightTriangle
    ::evaluatePdf(const Real2 &p)
{
  // the triangle has an area of 1/2
  return Real(1) / Real(0.5);
} // end UnitSquareToIsoscelesRightTriangle::evaluatePdf()

