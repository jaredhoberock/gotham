/*! \file Material.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of Material class.
 */
#include "Material.h"
#include <iostream>
#include "ScatteringDistributionFunction.h"
#include "../api/ShaderApi.h"

Material
  ::~Material(void)
{
  ;
} // end Material::~Material()

ScatteringDistributionFunction *Material
  ::evaluateScattering(const DifferentialGeometry &dg) const
{
  return ShaderApi::null();
} // end Material::evaluate()

ScatteringDistributionFunction *Material
  ::evaluateEmission(const DifferentialGeometry &dg) const
{
  return ShaderApi::null();
} // end Material::evaluateEmission()

ScatteringDistributionFunction *Material
  ::evaluateSensor(const DifferentialGeometry &dg) const
{
  return ShaderApi::null();
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

