/*! \file UnitSquareToCosineHemisphere.inl
 *  \author Jared Hoberock
 *  \brief Inline file for UnitSquareToCosineHemisphere.h.
 */

#include "UnitSquareToCosineHemisphere.h"
#include "UnitSquareToConcentricDisk.h"

#include <math.h>

#ifndef PI
#define PI 3.14159265f
#endif // PI

#ifndef INV_PI
#define INV_PI (1.0f / PI)
#endif // INV_PI

template<typename Real, typename Real3>
  void UnitSquareToCosineHemisphere<Real,Real3>
    ::evaluate(const Real &u, const Real &v,
               Real &x, Real &y, Real &z,
               Real &pdf)
{
  // sample the disk
  UnitSquareToConcentricDisk<Real,Real3>::evaluate(u,v,x,y);

  // its important for z to never be exactly zero
  // because then the pdf == 0.0, which is supposed to
  // be impossible
  z = Real(1) - x*x - y*y;
  if(z < Real(0.000005)) z = Real(0.000005);
  z = sqrtf(z);

  pdf = evaluatePdf(x,y,z);
} // end UnitSquareToCosineHemisphere<Real,Real3>::evaluate()

template<typename Real, typename Real3>
  void UnitSquareToCosineHemisphere<Real,Real3>
    ::evaluate(const Real &u,
               const Real &v,
               Real3 &p,
               Real *pdf)
{
  // sample the disk
  UnitSquareToConcentricDisk<Real,Real3>::evaluate(u,v,p);

  // its important for z to never be exactly zero
  // because then the pdf == 0.0, which is supposed to
  // be impossible
  p[2] = Real(1) - p[0]*p[0] - p[1]*p[1];
  p[2] = (p[2] > Real(0.000005)) ? p[2] : Real(0.000005);
  p[2] = sqrt(p[2]);

  if(pdf != 0)
  {
    *pdf = evaluatePdf(p);
  } // end if
} // end UnitSquareToHemisphere::evaluate()

template<typename Real, typename Real3>
  Real UnitSquareToCosineHemisphere<Real,Real3>
    ::evaluatePdf(const Real &x,
                  const Real &y,
                  const Real &z)
{
  return z * INV_PI;
} // end UnitSquareToCosineHemisphere<Real,Real3>::evaluatePdf()

template<typename Real, typename Real3>
  Real UnitSquareToCosineHemisphere<Real,Real3>
    ::evaluatePdf(const Real3 &p)
{
  return evaluatePdf(p[0],p[1],p[2]);
} // end UnitSquareToCosineHemisphere<Real,Real3>::evaluatePdf()

template<typename Real, typename Real3>
  void UnitSquareToCosineHemisphere<Real,Real3>
     ::inverse(const Real3 &p,
               Real &u,
               Real &v)
{
  UnitSquareToConcentricDisk<Real,Real3>::inverse(p,u,v);
} // end UnitSquareToCosineHemisphere<Real,Real3>::inverse()

template<typename Real, typename Real3>
  Real UnitSquareToCosineHemisphere<Real,Real3>
    ::normalizationConstant(void)
{
  return Real(1.5) * PI;
} // end UnitSquareToCosineHemisphere<Real,Real3>::normalizationConstant()

