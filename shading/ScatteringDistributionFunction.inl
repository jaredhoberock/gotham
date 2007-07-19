/*! \file ScatteringDistributionFunction.inl
 *  \author Jared Hoberock
 *  \brief Inline file for ScatteringDistributionFunction.h.
 */

#include "ScatteringDistributionFunction.h"

bool ScatteringDistributionFunction
  ::areSameHemisphere(const Vector3 &wo,
                      const Normal &n,
                      const Vector3 &wi)
{
  return wo.dot(n) * wi.dot(n) > 0;
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

