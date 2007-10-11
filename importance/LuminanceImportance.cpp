/*! \file LuminanceImportance.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of LuminanceImportance class.
 */

#include "LuminanceImportance.h"

float LuminanceImportance
  ::evaluate(const PathSampler::HyperPoint &x,
             const Path &xPath,
             const std::vector<PathSampler::Result> &results)
{
  return evaluateImportance(x,xPath,results);
} // end LuminanceImportance::evaluate()

float LuminanceImportance
  ::evaluateImportance(const PathSampler::HyperPoint &x,
                       const Path &xPath,
                       const std::vector<PathSampler::Result> &results)
{
  Spectrum L(0,0,0);
  for(std::vector<PathSampler::Result>::const_iterator r = results.begin();
      r != results.end();
      ++r)
  {
    //L += r->mThroughput * r->mWeight;
    L += r->mThroughput * r->mWeight / r->mPdf;
  } // end for r

  return L.luminance();
} // end LuminanceImportance::evaluateImportance()

