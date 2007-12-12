/*! \file ScatteringDistributionFunction.inl
 *  \author Jared Hoberock
 *  \brief Inline file for ScatteringDistributionFunction.h.
 */

#include "ScatteringDistributionFunction.h"

ScatteringDistributionFunction
  ::~ScatteringDistributionFunction(void)
{
  ;
} // end ScatteringDistributionFunction::~ScatteringDistributionFunction()

bool ScatteringDistributionFunction
  ::areSameHemisphere(const Vector3 &wo,
                      const Normal &n,
                      const Vector3 &wi)
{
  return areSameHemisphere(wo.dot(n), wi.dot(n));
} // end ScatteringDistributionFunction::areSameHemisphere()

bool ScatteringDistributionFunction
  ::areSameHemisphere(const float coso,
                      const float cosi)
{
  return (coso > 0) == (cosi > 0); 
} // end ScatteringDistributionFunction::areSameHemisphere()

Spectrum ScatteringDistributionFunction
  ::operator()(const Vector3 &wo,
               const DifferentialGeometry &dg,
               const Vector3 &wi) const
{
  return evaluate(wo,dg,wi);
} // end ScatteringDistributionFunction::operator()()

Spectrum ScatteringDistributionFunction
  ::operator()(const Vector3 &w,
               const DifferentialGeometry &dg) const
{
  return evaluate(w,dg);
} // end ScatteringDistributionFunction::operator()()

