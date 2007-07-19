/*! \file ScatteringDistributionFunction.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of ScatteringDistributionFunction class.
 */

#include "ScatteringDistributionFunction.h"
#include "../geometry/Mappings.h"

Spectrum ScatteringDistributionFunction
  ::evaluate(const Vector3 &wo,
             const DifferentialGeometry &dg,
             const Vector3 &wi) const
{
  return Spectrum::black();
} // end ScatteringDistributionFunction::evaluate()

Spectrum ScatteringDistributionFunction
  ::evaluate(const Vector3 &w,
             const DifferentialGeometry &dg) const
{
  return Spectrum::black();
} // end ScatteringDistributionFunction::evaluate()

void *ScatteringDistributionFunction
  ::operator new(unsigned int size)
{
  return mPool.malloc();
} // end ScatteringDistributionFunction::operator new()

float ScatteringDistributionFunction
  ::evaluatePdf(const Vector3 &wo,
                const DifferentialGeometry &dg,
                const Vector3 &wi) const
{
  return std::max(0.0f, dg.getNormal().dot(wi)) * INV_PI;
} // end ScatteringDistributionFunction::evaluatePdf()

Spectrum ScatteringDistributionFunction
  ::sample(const Vector3 &wo,
           const DifferentialGeometry &dg,
           const float u0,
           const float u1,
           const float u2,
           Vector3 &wi,
           float &pdf) const
{
  Mappings::unitSquareToCosineHemisphere(u0, u1, dg.getPointPartials()[0], dg.getPointPartials()[1], dg.getNormal(), wi, pdf);
  return evaluate(wo, dg, wi);
} // end ScatteringDistributionFunction::sample()

Spectrum ScatteringDistributionFunction
  ::sample(const DifferentialGeometry &dg,
           const float u0,
           const float u1,
           const float u2,
           Vector3 &w,
           float &pdf) const
{
  Mappings::unitSquareToCosineHemisphere(u0, u1, dg.getPointPartials()[0], dg.getPointPartials()[1], dg.getNormal(), w, pdf);
  return evaluate(w, dg);
} // end ScatteringDistributionFunction::sample()

float ScatteringDistributionFunction
  ::evaluatePdf(const Vector &w,
                const DifferentialGeometry &dg) const
{
  return std::max(0.0f, dg.getNormal().dot(w)) * INV_PI;
} // end ScatteringDistributionFunctionr::evaluatePdf()

