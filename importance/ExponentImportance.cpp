/*! \file ExponentImportance.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of ExponentImportance class.
 */

#include "ExponentImportance.h"

ExponentImportance
  ::ExponentImportance(const float k)
    :Parent(),mExponent(k)
{
  ;
} // end ExponentImportance::ExponentImportance()

float ExponentImportance
  ::evaluate(const PathSampler::HyperPoint &x,
             const Path &xPath,
             const std::vector<PathSampler::Result> &results)
{
  //return powf(Parent::evaluate(x,xPath,results), mExponent);

  Spectrum L(0,0,0);
  for(std::vector<PathSampler::Result>::const_iterator r = results.begin();
      r != results.end();
      ++r)
  {
    L += powf(r->mThroughput.luminance() * r->mWeight, mExponent) / r->mPdf;
  } // end for r

  return L.luminance();
} // end ExponentImportance::evaluate()

