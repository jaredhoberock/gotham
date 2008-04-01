/*! \file TargetImportance.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of TargetImportance class.
 */

#include "TargetImportance.h"

TargetImportance
  ::TargetImportance(const RandomAccessFilm &estimate,
                     const RandomAccessFilm &target,
                     const std::string &filename)
    :Parent(),
     mEstimateImportance(estimate),
     mTarget(target),
     mTargetFilename(filename)
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
               const boost::shared_ptr<ShadingContext> &context,
               const boost::shared_ptr<PathMutator> &mutator,
               MetropolisRenderer &renderer)
{
  mEstimateImportance.preprocess(r,scene,context,mutator,renderer);
  Parent::preprocess(r,scene,context,mutator,renderer);
} // end TargetImportance::preprocess()

void TargetImportance
  ::postprocess(void)
{
  Parent::postprocess();

  if(mTargetFilename != "")
  {
    // scale target so that it has a mean of 0.5
    float s = 0.5f / mTarget.computeMean().luminance();
    mTarget.scale(Spectrum(s,s,s));

    // write to file
    mTarget.writeEXR(mTargetFilename.c_str());
  } // end if
} // end TargetImportance::postprocess()

