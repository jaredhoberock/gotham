/*! \file KelemenMutator.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of KelemenMutator class.
 */

#include "KelemenMutator.h"
using namespace boost;

KelemenMutator
  ::KelemenMutator(void)
    :Parent()
{
  ;
} // end KelemenMutator::KelemenMutator()

KelemenMutator
  ::KelemenMutator(const shared_ptr<RandomSequence> &sequence,
                   const float p,
                   const shared_ptr<PathSampler> &s)
   :Parent(sequence, s)
{
  setLargeStepProbability(p);
} // end KelemenMutator::KelemenMutator()

KelemenMutator
  ::KelemenMutator(const float p,
                   const shared_ptr<PathSampler> &s)
   :Parent(s)
{
  setLargeStepProbability(p);
} // end KelemenMutator::KelemenMutator()

void KelemenMutator
  ::setLargeStepProbability(const float p)
{
  mLargeStepProbability = std::min(1.0f, p);
} // end KelemenMutator::setLargeStepProbability()

void KelemenMutator
  ::largeStep(PathSampler::HyperPoint &y)
{
  // generate a new HyperPoint
  PathSampler::constructHyperPoint(*mRandomSequence, y);
} // end KelemenMutator::largeStep()

bool KelemenMutator
  ::largeStep(PathSampler::HyperPoint &y,
              Path &b)
{
  largeStep(y);

  // sample a new Path
  return mSampler->constructPath(*mScene, *mShadingContext, y, b);
} // end KelemenMutator::largeStep()

bool KelemenMutator
  ::smallStep(const PathSampler::HyperPoint &x,
              const Path &a,
              PathSampler::HyperPoint &y,
              Path &b) const
{
  // mutate x into y
  smallStep(x,y);

  // sample a new Path
  return mSampler->constructPath(*mScene, *mShadingContext, y, b);
} // end KelemenMutator::smallStep()

void KelemenMutator
  ::smallStep(const PathSampler::HyperPoint &x,
              PathSampler::HyperPoint &y) const
{
  // mutate each coordinate of each hypercoordinate
  for(size_t i = 0; i < y.size(); ++i)
  {
    for(size_t j = 0; j < y[i].size(); ++j)
    {
      y[i][j] = smallStep(x[i][j]);
    } // end for j
  } // end for i
} // end KelemenMutator::smallStep()

float KelemenMutator
  ::smallStep(const float x) const
{
  float result;

  // these are the parameters Veach recommends for
  // film plane samples; Kelemen uses them in general
  const static float s1 = 1./1024, s2 = 1./64;
  const static float ratio = s2/s1;
  const static float c = logf(ratio);
  const static float epsilon = 1e-7f;

  float u = (*mRandomSequence)();
  float dv = s2 * expf(-c * u);
  float u1 = (*mRandomSequence)();

  if(u1 < 0.5f)
  {
    // Note that generating a value of 1.0f is an error
    result = x + dv; if (result >= 1.0f) result -= 1.0f;
  } // end if
  else
  {
    result = x - dv;

    if(result < 0.0f)
    {
      result += 1.0f;
    } // end if
  } // end else

  result = std::max(0.0f,result);
  result = std::min(1.0f - epsilon, result);

  return result;
} // end KelemenMutator::smallStep()

int KelemenMutator
  ::mutate(const PathSampler::HyperPoint &x,
           const Path &a,
           PathSampler::HyperPoint &y,
           Path &b)
{
  int result = -1;
  if((*mRandomSequence)() < mLargeStepProbability)
  {
    if(largeStep(y, b)) result = 1;
  } // end if
  else
  {
    if(smallStep(x, a, y, b)) result = 0;
  } // end else

  return result;
} // end KelemenMutator::mutate()

float KelemenMutator
  ::evaluateTransitionRatio(const unsigned int which,
                            const PathSampler::HyperPoint &x,
                            const Path &a,
                            const float ix,
                            const PathSampler::HyperPoint &y,
                            const Path &b,
                            const float iy) const
{
  float result = 1.0f;
  //if(which == 1)
  //{
  //  if(a.getSubpathLengths()[0] > 0)
  //  {
  //    result *= a[a.getSubpathLengths()[0]-1].mAccumulatedPdf;
  //  } // end if
  //  if(a.getSubpathLengths()[1] > 0)
  //  {
  //    result *= a[a.getSubpathLengths()[0]].mAccumulatedPdf;
  //  } // end if

  //  result *= a.getTerminationProbabilities().product();

  //  if(b.getSubpathLengths()[0] > 0)
  //  {
  //    result /= b[b.getSubpathLengths()[0]-1].mAccumulatedPdf;
  //  } // end if
  //  if(b.getSubpathLengths()[1] > 0)
  //  {
  //    result /= b[b.getSubpathLengths()[0]].mAccumulatedPdf;
  //  } // end if

  //  result /= b.getTerminationProbabilities().product();
  //} // end if
  //else
  //{
  //  result = 1.0f;
  //} // end else

  return result;
} // end KelemenMutator::evaluateTransitionRatio()

Spectrum KelemenMutator
  ::evaluate(const Path &x,
             std::vector<PathSampler::Result> &results)
{
  mSampler->evaluate(*mScene, x, results);

  // XXX TODO kill all of this - it's not used
  // XXX we may wish to return the maximum over results rather than the sum
  Spectrum L(0,0,0);
  for(std::vector<PathSampler::Result>::const_iterator r = results.begin();
      r != results.end();
      ++r)
  {
    //L += r->mThroughput * r->mWeight;
    L += r->mThroughput * r->mWeight / r->mPdf;

    assert(r->mEyeLength + r->mLightLength <= x.getSubpathLengths().sum());
  } // end for r

  return L;
} // end KelemenMutator::evaluate()

float KelemenMutator
  ::getLargeStepProbability(void) const
{
  return mLargeStepProbability;
} // end KelemenMutator::getLargeStepProbability()

