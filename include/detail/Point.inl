/*! \file Point.inl
 *  \author Jared Hoberock
 *  \brief Inline file for Point.h.
 */

#include <limits>

Point::Point(void):Parent()
{
  ;
} // end Point::Point()

Point::Point(const Parent &p):Parent(p)
{
  ;
} // end Point::Point()

Point
  ::Point(const float x, const float y, const float z)
    :Parent(x,y,z)
{
  ;
} // end Point::Point()

