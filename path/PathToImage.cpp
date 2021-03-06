/*! \file PathToImage.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of PathToImage class.
 */

#include "PathToImage.h"

void PathToImage
  ::evaluate(const PathSampler::Result &r,
             const PathSampler::HyperPoint &x,
             const Path &xPath,
             float &u,
             float &v) const
{
  if(r.mEyeLength > 1)
  {
    // the first coordinate naturally corresponds to (u,v)
    u = x[0][0];
    v = x[0][1];
  } // end if
  else if(r.mEyeLength == 0)
  {
    unsigned int endOfLightPath = xPath.getSubpathLengths().sum() - r.mLightLength;
    ::Vector w = xPath[endOfLightPath+1].mDg.getPoint();
    w -= xPath[endOfLightPath].mDg.getPoint();
    xPath[endOfLightPath].mSensor->invert(w, xPath[endOfLightPath].mDg,
                                          u, v);
  } // end if
  else if(r.mEyeLength == 1)
  {
    // we need to invert the sensor function
    unsigned int endOfLightPath = xPath.getSubpathLengths().sum() - r.mLightLength;
    ::Vector w = xPath[endOfLightPath].mDg.getPoint();
    w -= xPath[0].mDg.getPoint();
    xPath[0].mSensor->invert(w, xPath[0].mDg,
                             u, v);
  } // end if
} // end PathToImage::evaluate()

void PathToImage
  ::operator()(const PathSampler::Result &r,
               const PathSampler::HyperPoint &x,
               const Path &xPath,
               float &u,
               float &v) const
{
  return evaluate(r,x,xPath,u,v);
} // end PathToImage::operator()()
