/*! \file UnitSquareToAnisotropicLobe.inl
 *  \author Jared Hoberock
 *  \brief Inline file for UnitSquareToAnisotropicLobe.h.
 */

#include "UnitSquareToAnisotropicLobe.h"

#ifndef PI
#define PI 3.14159265f
#endif // PI

#ifndef TWO_PI
#define TWO_PI 6.28318531f
#endif // TWO_PI

#ifndef INV_TWOPI
#define INV_TWOPI 0.159154943f
#endif // INV_TWOPI

template<typename Real, typename Real3>
  void UnitSquareToAnisotropicLobe<Real,Real3>
    ::sampleFirstQuadrant(const Real &u,
                          const Real &v,
                          const Real &nu,
                          const Real &nv,
                          Real &phi,
                          Real &costheta)
{
  if(nu == nv)
  {
    phi = PI * u * Real(0.5);
  } // end if
  else
  {
    phi = atanf(sqrtf((nu+Real(1)) * (nv+Real(1))) * tanf(PI * u * Real(0.5)));
  } // end else

  Real cosphi = cosf(phi), sinphi = sinf(phi);
  costheta = powf(v, Real(1) / (nu * cosphi * cosphi + nv * sinphi + sinphi + Real(1)));
} // end UnitSquareToAnisotropicLobe::sampleFirstQuadrant()

template<typename Real, typename Real3>
  void UnitSquareToAnisotropicLobe<Real,Real3>
    ::evaluate(Real u,
               Real v,
               Real nu,
               Real nv,
               Real3 &p,
               Real *pdf)
{
  Real phi, costheta;
  Real factor, offset;
  if(u < Real(0.25))
  {
    u = Real(4) * u;
    factor = Real(1);
    offset = 0;
  } // end if
  else if(u < Real(0.5))
  {
    u = Real(4) * (Real(0.5) - u);
    factor = Real(-1);
    offset = Real(PI);
  } // end else if
  else if(u < Real(0.75))
  {
    u = Real(4) * (u - Real(0.5));
    factor = Real(1);
    offset = Real(PI);
  } // end else if
  else
  {
    u = Real(4) * (Real(1) - u);
    factor = Real(-1);
    offset = Real(TWO_PI);
  } // end else

  // sample the first quadrant
  sampleFirstQuadrant(u, v, nu, nv, phi, costheta);

  // transform the result
  phi = factor * phi + offset;

  float sintheta = Real(1) - costheta*costheta;
  sintheta = (sintheta > 0) ? sintheta : 0;
  p.x = sintheta * cosf(phi);
  p.y = sintheta * sinf(phi);
  p.z = fabsf(costheta);

  if(pdf) *pdf = evaluatePdf(p, nu, nv);
} // end UnitSquareToAnisotropicLobe::evaluate()

template<typename Real, typename Real3>
  Real UnitSquareToAnisotropicLobe<Real,Real3>
    ::evaluatePdf(const Real3 &p,
                  const Real &nu,
                  const Real &nv)
{
  float costheta = fabsf(p.z);
  float e = (nu * p.x * p.x + nv * p.y * p.y) / (Real(1) - costheta * costheta);
  return sqrtf((nu+Real(2)) * (nv+Real(2))) * INV_TWOPI * powf(costheta, e);
} // end UnitSquareToAnisotropicLobe::evaluatePdf()

