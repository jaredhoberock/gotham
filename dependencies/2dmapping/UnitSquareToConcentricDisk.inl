/*! \file UnitSquareToConcentricDisk.inl
 *  \author Jared Hoberock
 *  \brief Inline file for UnitSquareToConcentricDisk.h.
 */

#include "UnitSquareToConcentricDisk.h"

#ifndef PI
#define PI 3.14159265f
#endif // PI

#ifndef INV_PI
#define INV_PI (1.0f / PI)
#endif // INV_PI

template<typename Real, typename Real3>
  void UnitSquareToConcentricDisk<Real,Real3>
    ::evaluate(const Real &u,
               const Real &v,
               Real &x, Real &y,
               Real &pdf)
{
  evaluate(u,v,x,y);
  pdf = INV_PI;
} // end UnitSquareToConcentricDisk::evaluate()

template<typename Real, typename Real2>
  void UnitSquareToConcentricDisk<Real,Real2>
    ::evaluate(const Real &u,
               const Real &v,
               Real &x, Real &y)
{
  Real r, theta;

  // map uniform random variables to [-1, 1) x [-1, 1)
  Real sx = Real(2) * u - Real(1);
  Real sy = Real(2) * v - Real(1);

  // map square to (r, theta)

  // handle degeneracy at the origin
  if(sx == 0 && sy == 0)
  {
    x = y = 0;
    return;
  } // end if

  if(sx >= -sy)
  {
    if(sx > sy)
    {
      // first region of disk
      r = sx;
      if(sy > 0)
      {
        theta = sy / r;
      } // end if
      else
      {
        theta = Real(8) + sy / r;
      } // end else
    } // end if
    else
    {
      // second region of disk
      r = sy;
      theta = Real(2) - sx/r;
    } // end else
  } // end if
  else
  {
    if(sx <= sy)
    {
      // third region of disk
      r = -sx;
      theta = Real(4) - sy/r;
    } // end if
    else
    {
      // fourth region of disk
      r = -sy;
      theta = Real(6) + sx/r;
    } // end else
  } // end else

  theta *= Real(0.25) * PI;

  // XXX we need a way of identifying
  //     which precision of trig function
  //     to use
  x = r*cosf(theta);
  y = r*sinf(theta);

  return;
} // end UnitSquareToConcentricDisk::evaluate()

template<typename Real, typename Real2>
  void UnitSquareToConcentricDisk<Real,Real2>
    ::evaluate(const Real &u,
               const Real &v,
               Real2 &p,
               Real *pdf)
{
  evaluate(u,v,p[0],p[1]);
  if(pdf != 0) *pdf = INV_PI;

  return;
} // end UnitSquareToConcentricDisk::evaluate()

template<typename Real, typename Real2>
  void UnitSquareToConcentricDisk<Real,Real2>
    ::inverse(const Real2 &p,
              Real &u,
              Real &v)
{
  Real r = sqrt(p[0]*p[0] + p[1]*p[1]);
  if(r == 0)
  {
    u = v = 0.5f;
    return;
  } // end if

  Real theta = atan2(p[1], p[0]);

  // atan2 returns theta in [-pi, +pi]
  // transform theta to [-4, +4]
  theta *= (4.0f * INV_PI);
  // transform theta to [0, +8]
  theta += (theta < 0) ? Real(8) : Real(0);
  Real sx, sy;

  // figure out which quadrant of the disk we're in
  sx = sy = Real(10);
  if(theta > 7.0f)
  {
    sx = r;
    sy = r*(theta - 8.0f);
  } // end if
  else if(theta < 1.0f)
  {
    sx = r;
    sy = r*theta;
  } // end else if
  else if(theta < 2.0f)
  {
    sy = r;
    sx = r*(2.0f - theta);
  } // end else if
  else if(theta < 3.0f)
  {
    sy = r;
    sx = r*(2.0f - theta);
  } // end if
  else if(theta < 5.0f)
  {
    sx = -r;
    sy = r*(4.0f - theta);
  } // end else if
  else
  {
    sy = -r;
    sx = r*(theta - 6.0f);
  } // end else if

  // map uniform random variables from [-1,1) to [0,1)
  u = (sx + 1.0f) / 2.0f;
  v = (sy + 1.0f) / 2.0f;
} // end UnitSquareToConcentricDisk::inverse()

