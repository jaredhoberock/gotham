/*! \file SeededMutator.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of SeededMutator class.
 */

#include "SeededMutator.h"
#include "../shading/ScatteringDistributionFunction.h"

SeededMutator
  ::SeededMutator(void)
    :Parent()
{
  ;
} // end SeededMutator::SeededMutator()

SeededMutator
  ::SeededMutator(const float p,
                  const boost::shared_ptr<PathSampler> &s,
                  const size_t n)
    :Parent(p,s),mNumRejectionSamples(n)
{
  ;
} // end SeededMutator::SeededMutator()

SeededMutator
  ::SeededMutator(const boost::shared_ptr<RandomSequence> &sequence,
                  const float p,
                  const boost::shared_ptr<PathSampler> &s,
                  const size_t n)
    :Parent(sequence,p,s),mNumRejectionSamples(n)
{
  ;
} // end SeededMutator::SeededMutator()

bool SeededMutator
  ::largeStep(PathSampler::HyperPoint &y,
              Path &b)
{
  if(mSeeds.empty()) return Parent::largeStep(y,b);

  float u = (*Parent::mRandomSequence)();
  size_t i = static_cast<size_t>(u * static_cast<float>(mSeeds.size()));

  y = mSeedPoints[i];

  // clone using the global pool
  mSeeds[i]->clone(b, ScatteringDistributionFunction::mPool);

  return true;
} // end SeededMutator::largeStep()

void SeededMutator
  ::preprocess(void)
{
  Parent::preprocess();

  // reserve the maximum number of scattering functions we should need
  // 3: one per sensor, scattering, and emission
  // Path::static_size: times the maximum number of PathVertices allowed
  // mNumRejectionSamples: times the maximum number of Paths we will find
  mLocalPool.reserve(3 * Path::static_size * mNumRejectionSamples);

  Path xPath;
  PathSampler::HyperPoint x;
  std::vector<PathSampler::Result> results;
  for(size_t i = 0; i < mNumRejectionSamples; ++i)
  {
    PathSampler::constructHyperPoint(*mRandomSequence, x);
    if(mSampler->constructPath(*mScene, x, xPath))
    {
      // evaluate the Path
      results.clear();
      mSampler->evaluate(*mScene, xPath, results);
      if(!results.empty())
      {
        mSeedPoints.push_back(x);

        mSeeds.push_back(new Path());

        // safely copy xPath into seed
        xPath.clone(*mSeeds.back(), mLocalPool);
      } // end if
    } // end if

    // free all integrands allocated in this sample
    ScatteringDistributionFunction::mPool.freeAll();
  } // end for i
} // end SeededMutator::preprocess()

void SeededMutator
  ::postprocess(void)
{
  Parent::postprocess();

  for(size_t i = 0; i < mSeeds.size(); ++i)
  {
    delete mSeeds[i];
  } // end for i

  mSeeds.clear();
  mSeedPoints.clear();
  mLocalPool.freeAll();
} // end SeededMutator::postprocess()
