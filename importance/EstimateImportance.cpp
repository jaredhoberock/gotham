/*! \file EstimateImportance.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of EstimateImportance class.
 */

#include "EstimateImportance.h"
#include "LuminanceImportance.h"
#include "MaxImportance.h"

EstimateImportance
  ::EstimateImportance(const RandomAccessFilm &estimate)
    :mMapToImage(),
     mEstimate(estimate)
{
  for(size_t y = 0; y < mEstimate.getHeight(); ++y)
  {
    for(size_t x = 0; x < mEstimate.getWidth(); ++x)
    {
      Spectrum &e = mEstimate.raster(x,y);
      e[0] = 1.0f / std::max(0.05f, e.luminance());
    } // end for x
  } // end for y
} // end EstimateImportance::EstimateImportance()

float EstimateImportance
  ::evaluate(const PathSampler::HyperPoint &x,
             const Path &xPath,
             const std::vector<PathSampler::Result> &results)
{
  float I = 0;
  for(std::vector<PathSampler::Result>::const_iterator r = results.begin();
      r != results.end();
      ++r)
  {
    I += evaluate(x,xPath,*r);
  } // end for r

  return I;
} // end EstimateImportance::operator()()

float EstimateImportance
  ::evaluate(const PathSampler::HyperPoint &x,
             const Path &xPath,
             const PathSampler::Result &r) const
{
  gpcpu::float2 pixel;

  // scale each result by its corresponding bucket
  mMapToImage(r, x, xPath, pixel[0], pixel[1]);

  return mEstimate.pixel(pixel[0], pixel[1])[0]
    * LuminanceImportance::evaluateImportance(x,xPath,r);
} // end EstimateImportance::evaluate()

