/*! \file Material.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of Material class.
 */

#include "../include/Material.h"
#include "../include/ShadingInterface.h"

Material
  ::~Material(void)
{
  ;
} // end Material::~Material()

ScatteringDistributionFunction *Material
  ::evaluateScattering(ShadingInterface &context, const DifferentialGeometry &dg) const
{
  return context.null();
} // end Material::evaluate()

ScatteringDistributionFunction *Material
  ::evaluateEmission(ShadingInterface &context, const DifferentialGeometry &dg) const
{
  return context.null();
} // end Material::evaluateEmission()

ScatteringDistributionFunction *Material
  ::evaluateSensor(ShadingInterface &context, const DifferentialGeometry &dg) const
{
  return context.null();
} // end Material::evaluateSensor()

const char *Material
  ::getName(void) const
{
  return "Material";
} // end Material::getName()

bool Material
  ::isEmitter(void) const
{
  return false;
} // end Material::isEmitter()

bool Material
  ::isSensor(void) const
{
  return false;
} // end Material::isSensor()

