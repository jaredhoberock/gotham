/*! \file UnitSquareToTriangle.inl
 *  \author Jared Hoberock
 *  \brief Inline file for UnitSquareToTriangle.h.
 */

#include "UnitSquareToTriangle.h"
#include "UnitSquareToIsoscelesRightTriangle.h"

template<typename Real, typename Point>
  void UnitSquareToTriangle
    ::evaluate(const Real &u, const Real &v,
               const Point &v0, const Point &v1, const Point &v2,
               Point &p,
               Real *pdf)
{
  Real b[2];
  UnitSquareToIsoscelesRightTriangle::evaluate(u,v,b);

  p = b[0] * v0 + b[1] * v1 + (Real(1) - b[0] - b[1]) * v2;

  if(pdf)
  {
    *pdf = evaluatePdf<Real,Point>(p,v0,v1,v2);
  } // end if
} // end UnitSquareToTriangle::evaluate()

template<typename Real, typename Point, typename Point2>
  void UnitSquareToTriangle
    ::evaluate(const Real &u, const Real &v,
               const Point &v0, const Point &v1, const Point &v2,
               Point &p, Point2 &b,
               Real *pdf)
{
  UnitSquareToIsoscelesRightTriangle::evaluate(u,v,b);

  p = b[0] * v0 + b[1] * v1 + (Real(1) - b[0] - b[1]) * v2;

  if(pdf)
  {
    *pdf = evaluatePdf<Real,Point>(p,v0,v1,v2);
  } // end if
} // end UnitSquareToTriangle::evaluate()

template<typename Point, typename Real>
  void UnitSquareToTriangle
    ::inverse(const Point &p,
              const Point &v0,
              const Point &v1,
              const Point &v2,
              Real &u, Real &v)
{
  std::cerr << "UnitSquareToTriangle::inverse(): Unimplemented method called!" << std::endl;
} // end UnitSquareToTriangle::inverse()

template<typename Real, typename Point>
  Real UnitSquareToTriangle
    ::evaluatePdf(const Point &p, const Point &v0, const Point &v1, const Point &v2)
{
  // compute the area of (v0,v1,v2)
  Real denom = Real(0.5) * (v1 - v0).cross(v2 - v0).norm();
  return Real(1) / denom;
} // end UnitSquareToTriangle::evaluatePdf()

