/*! \file DeltaDistributionFunction.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of DeltaDistributionFunction class.
 */

#include "DeltaDistributionFunction.h"

float DeltaDistributionFunction
  ::evaluatePdf(const Vector3 &wo,
                const DifferentialGeometry &dg,
                const Vector3 &wi) const
{
  return 0;
} // end DeltaDistributionFunction::evaluatePdf()

float DeltaDistributionFunction
  ::evaluatePdf(const Vector3 &wo,
                const DifferentialGeometry &dg,
                const Vector3 &wi,
                const bool delta,
                const ComponentIndex component) const
{
  return delta ? 1.0f : 0.0f;
} // end DeltaDistributionFunction::evaluatePdf()

bool DeltaDistributionFunction
  ::isSpecular(void) const
{
  return true;
} // end DeltaDistributionFunction::isSpecular()

