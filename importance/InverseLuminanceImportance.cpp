/*! \file InverseLuminanceImportance.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of InverseLuminanceImportance class.
 */

#include "InverseLuminanceImportance.h"

float InverseLuminanceImportance
  ::operator()(const PathSampler::HyperPoint &x,
               const Spectrum &f)
{
  if(f.isBlack()) return 0;

  return 1.0f / f.luminance();
} // end InverseLuminanceImportance::operator()()

