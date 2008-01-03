/*! \file MaxImportance.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of MaxImportance class.
 */

#include "MaxImportance.h"

float MaxImportance
  ::evaluate(const PathSampler::HyperPoint &x,
             const Path &xPath,
             const std::vector<PathSampler::Result> &results)
{
  return evaluateImportance(x,xPath,results);
} // end MaxImportance::evaluate()

float MaxImportance
  ::evaluateImportance(const PathSampler::HyperPoint &x,
                       const Path &xPath,
                       const std::vector<PathSampler::Result> &results)
{
  float I = 0;
  for(std::vector<PathSampler::Result>::const_iterator r = results.begin();
      r != results.end();
      ++r)
  {
    I += evaluateImportance(x,xPath,*r);
  } // end for r

  return I;
} // end MaxImportance::evaluateImportance()

float MaxImportance
  ::evaluateImportance(const PathSampler::HyperPoint &x,
                       const Path &xPath,
                       const PathSampler::Result &r)
{
  return r.mThroughput.maxElement() * r.mWeight / r.mPdf;
} // end MaxImportance::evaluateImportance()

