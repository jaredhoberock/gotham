/*! \file UnitSquareToPhongHemisphere.inl
 *  \author Jared Hoberock
 *  \brief Inline file for UnitSquareToPhongHemisphere.h.
 */

#include "UnitSquareToPhongHemisphere.h"
#include <math.h>

#ifndef PI
#define PI 3.14159265f
#endif // PI

#ifndef INV_PI
#define INV_PI (1.0f / PI)
#endif // INV_PI

#ifndef INV_TWOPI
#define INV_TWOPI (1.0f / (2.0f * PI))
#endif // INV_TWOPI

template<typename Real, typename Real3>
  void UnitSquareToPhongHemisphere
    ::evaluate(const Real &u,
               const Real &v,
               const Real &k,
               Real3 &p,
               Real *pdf)
{
  Real cosAlpha = pow(1.0f - u,Real(1.0) / (k + Real(1.0)));

  float sinAlpha = sqrt(std::max<Real>(0, Real(1.0) - cosAlpha*cosAlpha));
  p[0] = sinAlpha * cos(Real(2.0) * PI * v);
  p[1] = sinAlpha * sin(Real(2.0) * PI * v);
  p[2] = cosAlpha;

  if(pdf != 0)
  {
    *pdf = evaluatePdf(p, k);
  } // end if
} // end UnitSquareToPhongHemisphere::evaluate()

template<typename Real, typename Real3>
  Real UnitSquareToPhongHemisphere
    ::evaluatePdf(const Real3 &p, const Real &k)
{
  // from Walter et al, 2007, equation 30
  return (k + Real(2.0)) * pow(p[2], k) * INV_TWOPI;
} // end UnitSquareToPhongHemisphere::evaluatePdf()

template<typename Real3, typename Real>
  void UnitSquareToPhongHemisphere
    ::inverse(const Real3 &p,
              const Real &k,
              Real &u,
              Real &v)
{
  //std::cerr << "UnitSquareToPhongHemisphere::inverse(): Implement me!" << std::endl;
} // end UnitSquareToPhongHemisphere::inverse()

