/*! \file ThroughputLuminanceImportance.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of ThroughputLuminanceImportance class.
 */

#include "ThroughputLuminanceImportance.h"

float ThroughputLuminanceImportance
  ::evaluate(const PathSampler::HyperPoint &x,
             const Path &xPath,
             const std::vector<PathSampler::Result> &results)
{
  Spectrum L(0,0,0);
  for(std::vector<PathSampler::Result>::const_iterator r = results.begin();
      r != results.end();
      ++r)
  {
    L += r->mThroughput * r->mWeight;
  } // end for r

  return L.luminance();
} // end ThroughputLuminanceImportance::evaluate()

