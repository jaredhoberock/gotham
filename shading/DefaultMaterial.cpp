/*! \file DefaultMaterial.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of DefaultMaterial class.
 */

#include "DefaultMaterial.h"
#include "Lambertian.h"

const char *DefaultMaterial
  ::getName(void) const
{
  return "DefaultMaterial";
} // end DefaultMaterial::getName()

ScatteringDistributionFunction *DefaultMaterial
  ::evaluateScattering(const DifferentialGeometry &dg) const
{
  return new Lambertian(Spectrum::white());
} // end DefaultMaterial::evaluate()
