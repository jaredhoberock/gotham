/*! \file Vector.inl
 *  \author Jared Hoberock
 *  \brief Inline file for Vector.h.
 */

#include "Vector.h"

Vector
  ::Vector(void)
    :Parent()
{
  ;
} // end Vector::Vector()

Vector
  ::Vector(const float x, const float y, const float z)
    :Parent(x,y,z)
{
  ;
} // end Vector::Vector()

Vector
  ::Vector(const Parent &v)
    :Parent(v)
{
  ;
} // end Vector::Vector()

