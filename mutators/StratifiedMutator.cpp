/*! \file StratifiedMutator.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of StratifiedMutator class.
 */

#include "StratifiedMutator.h"
#include <stratifiedsequence/StratifiedSequence.h>
using namespace boost;
using namespace gpcpu;

StratifiedMutator
  ::StratifiedMutator(const float largeStepProbability,
                      const shared_ptr<PathSampler> &s,
                      const uint2 &strata)
    :Parent(largeStepProbability, s),mStrata(strata)
{
  mCurrentSample = 0;
} // end StratifiedMutator::StratifiedMutator()

void StratifiedMutator
  ::reset(void)
{
  mSampleLocations.resize(mStrata.product());

  // stratify in [0,1) x [0,1)
  StratifiedSequence sequence(0, 1.0f, 0, 1.0f,
                              mStrata[0], mStrata[1]);
  float2 uv; 
  size_t i = 0;
  while(sequence(uv[0],uv[1],(*mRandomSequence)(), (*mRandomSequence)()))
  {
    mSampleLocations[i] = uv;
    ++i;
  } // end while

  // shuffle locations
  for(size_t k = 0; k < 5; ++k)
    std::random_shuffle(mSampleLocations.begin(), mSampleLocations.end());

  mCurrentSample = 0;
} // end StratifiedMutator::reset()

void StratifiedMutator
  ::largeStep(PathSampler::HyperPoint &y)
{
  Parent::largeStep(y);

  // replace y's pixel location with a stratified sample
  if(mCurrentSample >= mSampleLocations.size())
  {
    // need more samples
    reset();
  } // end if

  y[0][0] = mSampleLocations[mCurrentSample][0];
  y[0][1] = mSampleLocations[mCurrentSample][1];

  ++mCurrentSample;
} // end StratifiedMutator::largeStep()

