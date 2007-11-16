/*! \file HierarchicalLuminanceOverVisits.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of HierarchicalLuminanceOverVisits class.
 */

#include "HierarchicalLuminanceOverVisits.h"
#include "../records/MipMappedRenderFilm.h"

HierarchicalLuminanceOverVisits
  ::HierarchicalLuminanceOverVisits(const bool doInterpolate,
                                    const float radius)
    :Parent(doInterpolate),mRadius(radius)
{
  ;
} // end HierarchicalLuminanceOverVisits::HierarchicalLuminanceOverVisits()

float HierarchicalLuminanceOverVisits
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

  // XXX DESIGN this should probably not derive from anything
  //            and keep a proper MipMapRenderFilm pointer as a member
  const MipMappedRenderFilm *image = static_cast<const MipMappedRenderFilm*>(mAcceptance);

  // map x & y to their respective image locations
  gpcpu::float2 xPixel, yPixel;
  PathToImage mapToImage;

  mapToImage(xResults[0], x, xPath, xPixel[0], xPixel[1]);
  mapToImage(yResults[0], y, yPath, yPixel[0], yPixel[1]);

  // find the coarsest miplevel where they have differing raster locations
  gpcpu::size2 xRaster, yRaster;
  gpcpu::int2 diff;
  int level = image->getNumMipLevels();
  unsigned int manhattanDistance;
  float euclideanDistance;
  do
  {
    --level;
    image->getMipLevel(level).getRasterPosition(xPixel[0],  xPixel[1],
                                                xRaster[0], xRaster[1]);
    image->getMipLevel(level).getRasterPosition(yPixel[0],  yPixel[1],
                                                yRaster[0], yRaster[1]);

    // compute the Manhattan distance
    manhattanDistance = abs((int)xRaster[0] - (int)yRaster[0])
                      + abs((int)xRaster[1] - (int)yRaster[1]);

    // compute the Euclidean distance
    diff[0] = (int)xRaster[0] - (int)yRaster[0];
    diff[1] = (int)xRaster[1] - (int)yRaster[1];
    diff *= diff;
    euclideanDistance = sqrtf((float)diff[0] + (float)diff[1]);
  } // end do
  // XXX BUG why the fuck doesn't this work???
  //while(xRaster == yRaster && level > 0);
  // this isn't quite what we want i don't think
  //while(xRaster[0] == yRaster[0]
  //      && xRaster[1] == yRaster[1]
  //      && level > 0);
  //while(manhattanDistance <= 2 && level > 0);
  while(euclideanDistance <= mRadius && level > 0);
  ++level;
  image->getMipLevel(level).getRasterPosition(xPixel[0],  xPixel[1],
                                              xRaster[0], xRaster[1]);
  image->getMipLevel(level).getRasterPosition(yPixel[0],  yPixel[1],
                                              yRaster[0], yRaster[1]);

  // shorthand
  const RenderFilm &mipLevel = image->getMipLevel(level);

  // start with luminance of the path
  float result = mLuminanceImportance(x,xPath,xResults);

  // divide by constant's normalization constant
  // rationale: dividing some integrand by a unitless
  // ratio makes no sense
  // instead, make the function unitless
  result *= mLuminanceImportance.getInvNormalizationConstant();

  // how many accepts has each pixel of the image received, on average?
  float totalAccepts = mipLevel.getSum()[0];
  //float totalAccepts = mAcceptance->getSum()[0];
  if(totalAccepts != 0)
  {
    float numPixels = mipLevel.getWidth() * mipLevel.getHeight();
    //float numPixels = mAcceptance->getWidth() * mAcceptance->getHeight();
    float avg = totalAccepts / numPixels;

    // look up the number of visits this point has received
    float visits;
    if(mDoInterpolate)
    {
      // interpolating visits makes for a somewhat smoother result
      visits = mipLevel.bilerp(xPixel[0], xPixel[1])[0];
    } // end if
    else
    {
      visits = mipLevel.pixel(xPixel[0], xPixel[1])[0];
    } // end else

    // we need to return the ratio of the number of times
    // accepted over the average acceptance rate
    // or something like that
    result *= (avg / visits);
  } // end if

  return result;
} // end HierarchicalLuminanceOverVisits::evaluate()

