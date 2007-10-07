/*! \file Material.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of Material class.
 */
#include "Material.h"
#include <iostream>
#include "ScatteringDistributionFunction.h"

Material
  ::~Material(void)
{
  ;
} // end Material::~Material()

ScatteringDistributionFunction *Material
  ::evaluateScattering(const DifferentialGeometry &dg) const
{
  return new ScatteringDistributionFunction();
} // end Material::evaluate()

ScatteringDistributionFunction *Material
  ::evaluateEmission(const DifferentialGeometry &dg) const
{
  return new ScatteringDistributionFunction();
} // end Material::evaluateEmission()

ScatteringDistributionFunction *Material
  ::evaluateSensor(const DifferentialGeometry &dg) const
{
  return new ScatteringDistributionFunction();
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

