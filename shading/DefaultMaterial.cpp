/*! \file DefaultMaterial.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of DefaultMaterial class.
 */

#include "DefaultMaterial.h"
#include "Lambertian.h"
#include "../api/ShaderApi.h"

const char *DefaultMaterial
  ::getName(void) const
{
  return "DefaultMaterial";
} // end DefaultMaterial::getName()

ScatteringDistributionFunction *DefaultMaterial
  ::evaluateScattering(const DifferentialGeometry &dg) const
{
  return ShaderApi::diffuse(Spectrum::white());
} // end DefaultMaterial::evaluate()

