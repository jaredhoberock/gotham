/*! \file BoundingBox.inl
 *  \author Jared Hoberock
 *  \brief Inline file for BoundingBox.h.
 */

#include <limits>
#include "BoundingBox.h"

BoundingBox::BoundingBox(void):Parent()
{
  ;
} // end BoundingBox::BoundingBox()

BoundingBox::BoundingBox(const Point &min, const Point &max):Parent(min,max)
{
  ;
} // end BoundingBox::BoundingBox()

