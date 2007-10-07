/*! \file LuminanceImportance.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of LuminanceImportance class.
 */

#include "LuminanceImportance.h"

float LuminanceImportance
  ::operator()(const PathSampler::HyperPoint &x,
               const Spectrum &f)
{
  return f.luminance();
} // end LuminanceImportance::operator()()

