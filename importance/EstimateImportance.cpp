/*! \file EstimateImportance.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of EstimateImportance class.
 */

#include "EstimateImportance.h"

EstimateImportance
  ::EstimateImportance(const boost::shared_ptr<RenderFilm> &estimate)
    :mEstimate(estimate),mMapToImage()
{
  ;
} // end EstimateImportance::EstimateImportance()

float EstimateImportance
  ::evaluate(const PathSampler::HyperPoint &x,
             const Path &xPath,
             const std::vector<PathSampler::Result> &results)
{
  const RenderFilm *estimate = mEstimate.get();
  Spectrum I(0,0,0);
  gpcpu::float2 pixel;
  for(std::vector<PathSampler::Result>::const_iterator r = results.begin();
      r != results.end();
      ++r)
  {
    // scale each result by its corresponding bucket
    mMapToImage(*r, x, xPath, pixel[0], pixel[1]);
    //I += (r->mThroughput * r->mWeight / r->mPdf) * mEstimate.element(bucket[0], bucket[1]);

    // XXX PERF remove these division and max operations
    I += (r->mThroughput * r->mWeight / r->mPdf) / std::max(0.05f, estimate->pixel(pixel[0], pixel[1]).luminance());
  } // end for r

  return I.luminance();
} // end EstimateImportance::operator()()

