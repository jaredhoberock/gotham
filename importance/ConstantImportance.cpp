/*! \file ConstantImportance.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of ConstantImportance class.
 */

#include "ConstantImportance.h"

float ConstantImportance
  ::evaluate(const PathSampler::HyperPoint &x,
             const Path &xPath,
             const std::vector<PathSampler::Result> &results)
{
  //if(results.empty()) return 0;
  //return 1.0;

  float result = 0;
  for(std::vector<PathSampler::Result>::const_iterator r = results.begin();
      r != results.end();
      ++r)
  {
    result += 1.0f / r->mPdf;
  } // end for r

  return result;
} // end ConstantImportance::operator()()
