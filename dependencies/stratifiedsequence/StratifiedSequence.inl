/*! \file StratifiedSequence.inl
 *  \author Jared Hoberock
 *  \brief Inline file for StratifiedSequence.h.
 */

#include "StratifiedSequence.h"

StratifiedSequence
  ::StratifiedSequence(void)
    :Parent()
{
  ;
} // end StratifiedSequence::StratifiedSequence()

StratifiedSequence
  ::StratifiedSequence(const float xStart, const float xEnd,
                       const float yStart, const float yEnd,
                       const size_t xStrata,
                       const size_t yStrata)
    :Parent(xStart, xEnd, yStart, yEnd)
{
  reset(xStart, xEnd, yStart, yEnd, xStrata, yStrata);
} // end StratifiedSequence::StratifiedSequence()

void StratifiedSequence
  ::reset(const float xStart, const float xEnd,
          const float yStart, const float yEnd,
          const size_t xStrata,
          const size_t yStrata)
{
  Parent::reset(xStart, xEnd, yStart, yEnd);

  mOrigin[0] = xStart;
  mOrigin[1] = yStart;

  mNumStrata[0] = xStrata;
  mNumStrata[1] = yStrata;

  mStrataSpacing[0] = (xEnd - xStart) / xStrata;
  mStrataSpacing[1] = (yEnd - yStart) / yStrata;

  reset();
} // end StratifiedSequence::reset()

void StratifiedSequence
  ::reset(void)
{
  // start off one spacing to the left of (xStart, yStart)
  mCurrentPoint[0] = mOrigin[0] - mStrataSpacing[0];
  mCurrentPoint[1] = mOrigin[1];

  // the idea here is to make it roll over to 0 on
  // the first advance
  mCurrentRaster[0] = std::numeric_limits<size_t>::max();
  mCurrentRaster[1] = 0;
} // end StratifiedSequence::reset()

StratifiedSequence
  ::~StratifiedSequence(void)
{
  ;
} // end StratifiedSequence::~StratifiedSequence()

bool StratifiedSequence
  ::advance(void)
{
  // advance the current point & raster position
  mCurrentPoint[0] += mStrataSpacing[0];
  ++mCurrentRaster[0];
  if(mCurrentRaster[0] == mNumStrata[0])
  {
    mCurrentPoint[0] = mOrigin[0];
    mCurrentRaster[0] = 0;
    ++mCurrentRaster[1];
    mCurrentPoint[1] += mStrataSpacing[1];
  } // end if

  // have we finished?
  return mCurrentRaster[1] < mNumStrata[1];
} // end StratifiedSequence::advance()

bool StratifiedSequence
  ::operator()(float &x, float &y)
{
  bool result = advance();

  x = mCurrentPoint[0];
  y = mCurrentPoint[1];

  return result;
} // end StratifiedSequence::operator()()

bool StratifiedSequence
  ::operator()(float &x, float &y,
               const float xJitter, const float yJitter)
{
  bool result = advance();

  x = mCurrentPoint[0] + xJitter*mStrataSpacing[0];
  y = mCurrentPoint[1] + yJitter*mStrataSpacing[1];

  return result;
} // end StratifiedSequence::operator()()

const size_t *StratifiedSequence
  ::getCurrentRaster(void) const
{
  return mCurrentRaster;
} // end StratifiedSequence::getCurrentRaster()

