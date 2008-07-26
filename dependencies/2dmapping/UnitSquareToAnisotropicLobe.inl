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

template<typename Real>
  void UnitSquareToAnisotropicLobe
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
  void UnitSquareToAnisotropicLobe
    ::evaluate(const Real &u,
               const Real &v,
               const Real &nu,
               const Real &nv,
               Real3 &p,
               Real *pdf)
{
  float phi, costheta;
  if(u < Real(0.25))
  {
    sampleFirstQuadrant(Real(4) * u, v, nu, nv, phi, costheta);
  } // end if
  else if(u < Real(0.5))
  {
    sampleFirstQuadrant(Real(4) * (Real(0.5) - u), v, nu, nv, phi, costheta);
    phi = PI - phi;
  } // end else if
  else if(u < Real(0.75))
  {
    sampleFirstQuadrant(Real(4) * (u - Real(0.5)), v, nu, nv, phi, costheta);
    phi += PI;
  } // end else if
  else
  {
    sampleFirstQuadrant(Real(4) * (Real(1) - u), v, nu, nv, phi, costheta);
    phi = TWO_PI - phi;
  } // end else

  float sintheta = Real(1) - costheta*costheta;
  sintheta = (sintheta > 0) ? sintheta : 0;
  p[0] = sintheta * cosf(phi);
  p[1] = sintheta * sinf(phi);
  p[2] = fabsf(costheta);

  if(pdf) *pdf = evaluatePdf(p, nu, nv);
} // end UnitSquareToAnisotropicLobe::evaluate()

template<typename Real, typename Real3>
  Real UnitSquareToAnisotropicLobe
    ::evaluatePdf(const Real3 &p,
                  const Real &nu,
                  const Real &nv)
{
  float costheta = fabsf(p[2]);
  float e = (nu * p[0] * p[0] + nv * p[1] * p[1]) / (Real(1) - costheta * costheta);
  return sqrtf((nu+Real(2)) * (nv+Real(2))) * INV_TWOPI * powf(costheta, e);
} // end UnitSquareToAnisotropicLobe::evaluatePdf()

