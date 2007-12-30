/*! \file TargetImportance.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of TargetImportance class.
 */

#include "TargetImportance.h"

TargetImportance
  ::TargetImportance(const RandomAccessFilm &estimate,
                     const RandomAccessFilm &target)
    :Parent(),
     mEstimateImportance(estimate),
     mTarget(target)
{
  ;
} // end TargetImportance::TargetImportance()

float TargetImportance
  ::evaluate(const PathSampler::HyperPoint &x,
             const Path &xPath,
             const std::vector<PathSampler::Result> &results)
{
  float i = 0;
  float I = 0;
  gpcpu::float2 pixel;
  for(std::vector<PathSampler::Result>::const_iterator r = results.begin();
      r != results.end();
      ++r)
  {
    // scale each result by its corresponding bucket
    mMapToImage(*r, x, xPath, pixel[0], pixel[1]);

    // take what the Parent would do
    i = mEstimateImportance.evaluate(x,xPath,*r);

    // now multiply by the target
    i *= mTarget.pixel(pixel[0],pixel[1]).luminance();

    I += i;
  } // end for r

  return I;
} // end EstimateImportance::operator()()

void TargetImportance
  ::preprocess(const boost::shared_ptr<RandomSequence> &r,
               const boost::shared_ptr<const Scene> &scene,
               const boost::shared_ptr<PathMutator> &mutator,
               MetropolisRenderer &renderer)
{
  mEstimateImportance.preprocess(r,scene,mutator,renderer);
  Parent::preprocess(r,scene,mutator,renderer);
} // end TargetImportance::preprocess()

