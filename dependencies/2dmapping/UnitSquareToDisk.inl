/*! \file UnitSquareToDisk.inl
 *  \author Jared Hoberock
 *  \brief Inline file for UnitSquareToDisk.h.
 */

#include "UnitSquareToDisk.h"

#ifndef PI
#define PI 3.14159265f
#endif // PI

#ifndef INV_PI
#define INV_PI (1.0f / PI)
#endif // INV_PI

template<typename Real>
  void UnitSquareToDisk
    ::evaluate(const Real &u, const Real &v,
               Real &x, Real &y,
               Real &pdf)
{
  evaluate(u,y,x,y);

  pdf = INV_PI;
} // end UnitSquareToDisk::evaluate()

template<typename Real>
  void UnitSquareToDisk
    ::evaluate(const Real &u, const Real &v,
               Real &x, Real &y)
{
  Real r = sqrt(u);
  Real theta = Real(2) * PI * v;

  // XXX we should have a way to decide which
  //     precision trig functions to use
  x = r * cosf(theta);
  y = r * sinf(theta);
} // end UnitSquareToDisk::evaluate()

template<typename Real, typename Real2>
  void UnitSquareToDisk
    ::evaluate(const Real &u,
               const Real &v,
               Real2 &p,
               Real *pdf)
{
  evaluate(u,v,p[0],p[1]);

  if(pdf != 0)
  {
    *pdf = INV_PI;
  } // end if
} // end UnitSquareToDisk::evaluate()

template<typename Real2, typename Real>
  void UnitSquareToDisk
    ::inverse(const Real2 &p,
              Real &u,
              Real &v)
{
  u = sqrt(p[0]*p[0] + p[1]*p[1]);
  Real theta = 0;
  if(u != 0)
  {
    theta = acos(p[0] / u);
  } // end if

  v = theta / (2.0f * PI);
} // end UnitSquareToDisk::inverse()

