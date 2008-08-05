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

const char *Material
  ::getSource(void) const
{
  return "";
} // end Material::getSource()

size_t Material
  ::getScatteringParametersSize(void) const
{
  return 0;
} // end Material::getScatteringParametersSize(void)

void Material
  ::getScatteringParameters(void *ptr) const
{
  ;
} // end Material::getScatteringParameters()

size_t Material
  ::getEmissionParametersSize(void) const
{
  return 0;
} // end Material::getEmissionParametersSize(void)

void Material
  ::getEmissionParameters(void *ptr) const
{
  ;
} // end Material::getEmissionParameters()

size_t Material
  ::getSensorParametersSize(void) const
{
  return 0;
} // end Material::getSensorParametersSize(void)

void Material
  ::getSensorParameters(void *ptr) const
{
  ;
} // end Material::getSensorParameters()

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

