/*! \file ConstantImportance.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of ConstantImportance class.
 */

#include "ConstantImportance.h"

float ConstantImportance
  ::operator()(const PathSampler::HyperPoint &x,
               const Spectrum &f)
{
  return 1.0f;
  //if(x[0][0] >= 0.5f && x[0][1] >= 0.5f) return 1000.0f;
  //return 0.0001f;
} // end ConstantImportance::operator()()

