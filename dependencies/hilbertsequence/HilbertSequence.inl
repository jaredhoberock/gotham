/*! \file HilbertSequence.inl
 *  \author Jared Hoberock
 *  \brief Inline file for HilbertSequence.h.
 */

#include "HilbertSequence.h"

HilbertSequence
  ::HilbertSequence(void)
    :Parent()
{
  ;
} // end HilbertSequence::HilbertSequence()

HilbertSequence
  ::HilbertSequence(const float xStart, const float xEnd,
                    const float yStart, const float yEnd,
                    const size_t xStrata,
                    const size_t yStrata)
    :Parent(xStart, xEnd, yStart, yEnd, xStrata, yStrata)
{
  reset(xStart, xEnd, yStart, yEnd, xStrata, yStrata);
} // end HilbertSequence::HilbertSequence()

void HilbertSequence
  ::reset(const float xStart, const float xEnd,
          const float yStart, const float yEnd,
          const size_t xStrata,
          const size_t yStrata)
{
  Parent::reset(xStart, xEnd, yStart, yEnd, xStrata, yStrata);
  reset();
} // end HilbertSequence::reset()

void HilbertSequence
  ::reset(void)
{
  mBlockStack.clear();
  mBlockStack.push_back(gpcpu::uint4(0, mNumStrata[0], 0, mNumStrata[1]));
} // end HilbertSequence::reset()

bool HilbertSequence
  ::advance(void)
{
  if(!mWalk(mCurrentRaster[0], mCurrentRaster[1]))
  {
    // is there more work to do?
    if(mBlockStack.empty()) return false;

    gpcpu::uint4 work = mBlockStack.back();
    mBlockStack.pop_back();

    // cut up the work into three blocks
    gpcpu::uint2 workStart = gpcpu::uint2(work[0], work[2]);

    size_t workWidth = work[1];
    size_t workHeight = work[3];

    // find the largest power of two that is smaller than both workWidth & workHeight
    size_t newSize;
    size_t smaller = std::min(workWidth,workHeight);
    // XXX fix this grossness
    size_t powerOf2 = static_cast<unsigned int>(floor(logf(static_cast<float>(smaller))/logf(2.0f)));

    newSize = 1<<powerOf2;

    // add up to two blocks to the work stack:
    //
    // ------------------------
    // |          |           |
    // |  block2  |           |
    // |          |           |
    // |-----------           |
    // |          |           |
    // | next     |   block 1 |
    // | work     |           |
    // |          |           |
    // |-----------------------
    //
    if(newSize < workWidth)
    {
      mBlockStack.push_back(gpcpu::uint4(workStart[0] + newSize,
                                         workWidth - newSize,
                                         workStart[1],
                                         workHeight));
    } // end if

    if(newSize < workHeight)
    {
      mBlockStack.push_back(gpcpu::uint4(workStart[0],
                                         newSize,
                                         workStart[1] + newSize,
                                         workHeight - newSize));
    } // end if

    mWalk.init(powerOf2, workStart[0], workStart[1]);
    mWalk(mCurrentRaster[0], mCurrentRaster[1]);
  } // end if

  mCurrentPoint[0] = mOrigin[0] + mStrataSpacing[0] * static_cast<float>(mCurrentRaster[0]);
  mCurrentPoint[1] = mOrigin[1] + mStrataSpacing[1] * static_cast<float>(mCurrentRaster[1]);

  return true;
} // end HilbertSequence::advance()

