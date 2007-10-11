/*! \file InverseLuminanceImportance.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of InverseLuminanceImportance class.
 */

#include "InverseLuminanceImportance.h"
#include "LuminanceImportance.h"

float InverseLuminanceImportance
  ::evaluate(const PathSampler::HyperPoint &x,
             const Path &xPath,
             const std::vector<PathSampler::Result> &results)
{
  if(results.empty()) return 0;

  return 1.0f / LuminanceImportance::evaluateImportance(x,xPath,results);
} // end InverseLuminanceImportance::evaluate()

