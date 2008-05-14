/*! \file Quadratic.cpp
 *  \author Jared Hoberock 
 *  \brief Implementation of Quadratic class.
 */

#include "Quadratic.h"
#include <math.h>
#include <algorithm>

int Quadratic
  ::realRoots(float &x0, float &x1) const
{
  int result = 0;

  // are there roots?
  float denom = 2.0f*a();

  if(denom != 0.0f)
  {
    float root = b()*b() - 4.0f*a()*c();

    if(root == 0.0f)
    {
      // one root
      result = 1;

      x0 = x1 = -b() / denom;
    } // end if
    else if(root > 0.0f)
    {
      // two roots
      result = 2;

      root = sqrtf(root);

      x0 = (-b() - root) / denom;
      x1 = (-b() + root) / denom;

      if(x0 > x1) std::swap(x0,x1);
    } // end else if
  } // end if

  return result;
} // end Quadratic::realRoots()
