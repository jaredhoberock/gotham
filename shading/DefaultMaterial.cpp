/*! \file DefaultMaterial.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of DefaultMaterial class.
 */

#include "DefaultMaterial.h"
#include "Lambertian.h"
#include "../include/ShadingInterface.h"

const char *DefaultMaterial
  ::getName(void) const
{
  return "DefaultMaterial";
} // end DefaultMaterial::getName()

ScatteringDistributionFunction *DefaultMaterial
  ::evaluateScattering(ShadingInterface &context, const DifferentialGeometry &dg) const
{
  return context.diffuse(Spectrum::white());
} // end DefaultMaterial::evaluate()

