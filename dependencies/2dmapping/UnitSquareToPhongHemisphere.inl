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
  void UnitSquareToPhongHemisphere<Real,Real3>
    ::evaluate(const Real &u,
               const Real &v,
               const Real &k,
               Real3 &p,
               Real *pdf)
{
  Real cosAlpha = pow(Real(1) - u,Real(1) / (k + Real(1)));

  float sinAlpha = Real(1) - cosAlpha*cosAlpha;
  sinAlpha = sinAlpha > 0 ? sinAlpha : Real(0);
  sinAlpha = sqrtf(sinAlpha);

  p.x = sinAlpha * cosf(Real(2) * PI * v);
  p.y = sinAlpha * sinf(Real(2) * PI * v);
  p.z = cosAlpha;

  if(pdf != 0)
  {
    *pdf = evaluatePdf(p, k);
  } // end if
} // end UnitSquareToPhongHemisphere::evaluate()

template<typename Real, typename Real3>
  Real UnitSquareToPhongHemisphere<Real,Real3>
    ::evaluatePdf(const Real3 &p, const Real &k)
{
  // from Walter et al, 2007, equation 30
  return (k + Real(2)) * powf(p.z, k) * INV_TWOPI;
} // end UnitSquareToPhongHemisphere::evaluatePdf()

template<typename Real, typename Real3>
  void UnitSquareToPhongHemisphere<Real,Real3>
    ::inverse(const Real3 &p,
              const Real &k,
              Real &u,
              Real &v)
{
  //std::cerr << "UnitSquareToPhongHemisphere::inverse(): Implement me!" << std::endl;
} // end UnitSquareToPhongHemisphere::inverse()

