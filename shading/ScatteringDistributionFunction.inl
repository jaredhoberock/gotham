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

template<typename V3, typename N3>
  bool ScatteringDistributionFunction
    ::areSameHemisphere(const V3 &wo,
                        const N3 &n,
                        const V3 &wi)
{
  return areSameHemisphere(dot(wo,n), dot(wi,n));
} // end ScatteringDistributionFunction::areSameHemisphere()

bool ScatteringDistributionFunction
  ::areSameHemisphere(const float coso,
                      const float cosi)
{
  return (coso > 0) == (cosi > 0); 
} // end ScatteringDistributionFunction::areSameHemisphere()

Spectrum ScatteringDistributionFunction
  ::operator()(const Vector &wo,
               const DifferentialGeometry &dg,
               const Vector &wi) const
{
  return evaluate(wo,dg,wi);
} // end ScatteringDistributionFunction::operator()()

Spectrum ScatteringDistributionFunction
  ::operator()(const Vector &w,
               const DifferentialGeometry &dg) const
{
  return evaluate(w,dg);
} // end ScatteringDistributionFunction::operator()()

