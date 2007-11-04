/*! \file ManualImportance.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of ManualImportance class.
 */

#include "ManualImportance.h"
#include "../path/PathToImage.h"

float ManualImportance
  ::evaluate(const PathSampler::HyperPoint &x,
             const Path &xPath,
             const std::vector<PathSampler::Result> &results)
{
  float result = Parent::evaluate(x,xPath,results);
  if(result > 0)
  {
    gpcpu::float2 pixel;
    PathToImage mapToImage;
    mapToImage(results[0], x, xPath, pixel[0], pixel[1]);

    // if we lie on the right side of the image, scale x10
    if(pixel[0] >= 0.5f)
    {
      result *= 10.0f;
    } // end if
  } // end if

  return result;
} // end ManualImportance::evaluate()

