/*! \file ConditionalEqualVisitImportance.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of ConditionalEqualVisitImportance class.
 */

#include "ConditionalEqualVisitImportance.h"
#include "../path/PathToImage.h"
#include "../records/RenderFilm.h"

ConditionalEqualVisitImportance
  ::ConditionalEqualVisitImportance(const bool doInterpolate)
    :Parent(doInterpolate)
{
  ;
} // end ConditionalEqualVisitImportance::ConditionalEqualVisitImportance()

float ConditionalEqualVisitImportance
  ::evaluate(const PathSampler::HyperPoint &x,
             const Path &xPath,
             const std::vector<PathSampler::Result> &xResults,
             const PathSampler::HyperPoint &y,
             const Path &yPath,
             const std::vector<PathSampler::Result> &yResults)
{
  // XXX TODO PERF This implementation is pretty egregiously
  //          wasteful
  if(xResults.empty()) return 0;

  // map x & y to their respective pixels
  gpcpu::float2 xPixel, yPixel;
  PathToImage mapToImage;

  mapToImage(xResults[0], x, xPath, xPixel[0], xPixel[1]);
  mapToImage(yResults[0], y, yPath, yPixel[0], yPixel[1]);

  // find their raster positions
  gpcpu::size2 xRaster, yRaster;
  mAcceptance->getRasterPosition(xPixel[0], xPixel[1], xRaster[0], xRaster[1]);
  mAcceptance->getRasterPosition(yPixel[0], yPixel[1], yRaster[0], yRaster[1]);

  if(xRaster == yRaster)
  {
    // we're in the same bucket; proceed as normal
    float result = mLuminanceImportance(x,xPath,xResults);

    // divide by constant's normalization constant
    // rationale: dividing some integrand by a unitless
    // ratio makes no sense
    // instead, make the function unitless
    result *= mLuminanceImportance.getInvNormalizationConstant();
    return result;
  } // end if

  // we're in different buckets; compare acceptance rates
  float result = 1.0f;

  // they map to different locations, so return x's acceptance count
  // how many accepts has each pixel of the image received, on average?
  float totalAccepts = mAcceptance->getSum()[0];
  if(totalAccepts != 0)
  {
    float numPixels = mAcceptance->getWidth() * mAcceptance->getHeight();
    float avg = totalAccepts / numPixels;

    // look up the number of visits this point has received
    float visits;
    if(mDoInterpolate)
    {
      // interpolating visits makes for a somewhat smoother result
      visits = mAcceptance->bilerp(xPixel[0], xPixel[1])[0];
    } // end if
    else
    {
      visits = mAcceptance->pixel(xPixel[0], xPixel[1])[0];
    } // end else

    // we need to return the ratio of the number of times
    // accepted over the average acceptance rate
    // or something like that
    result *= (avg / visits);
  } // end if

  return result;
} // end ConditionalEqualVisitImportance::evaluate()

