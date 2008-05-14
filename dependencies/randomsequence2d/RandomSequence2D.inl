/*! \file RandomSequence2D.inl
 *  \author Jared Hoberock
 *  \brief Inline file for RandomSequence2D.h.
 */

#include "RandomSequence2D.h"

RandomSequence2D
  ::RandomSequence2D(void)
{
  ;
} // end RandomSequence2D::RandomSequence2D()

RandomSequence2D
  ::RandomSequence2D(const float xStart, const float xEnd,
                     const float yStart, const float yEnd)
    :mXStart(xStart),
     mDeltaX(xEnd - xStart),
     mYStart(yStart),
     mDeltaY(yEnd - yStart)
{
  ;
} // end RandomSequence2D::RandomSequence2D()

void RandomSequence2D
  ::reset(const float xStart, const float xEnd,
          const float yStart, const float yEnd)
{
  mXStart = xStart;
  mDeltaX = xEnd - xStart;
  mYStart = yStart;
  mDeltaY = yEnd - yStart;
} // end RandomSequence2D::reset()

RandomSequence2D
  ::~RandomSequence2D(void)
{
  ;
} // end RandomSequence2D::~RandomSequence2D()

bool RandomSequence2D
  ::operator()(float &x, float &y,
               float z0, float z1)
{
  x = mXStart + z0 * mDeltaX;
  y = mYStart + z1 * mDeltaY;
  return true;
} // end RandomSequence2D::operator()()

