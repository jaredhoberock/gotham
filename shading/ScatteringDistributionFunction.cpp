/*! \file ScatteringDistributionFunction.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of ScatteringDistributionFunction class.
 */

#include "ScatteringDistributionFunction.h"
#include "../geometry/Mappings.h"

Spectrum ScatteringDistributionFunction
  ::evaluate(const Vector &wo,
             const DifferentialGeometry &dg,
             const Vector &wi) const
{
  return Spectrum::black();
} // end ScatteringDistributionFunction::evaluate()

Spectrum ScatteringDistributionFunction
  ::evaluate(const Vector &w,
             const DifferentialGeometry &dg) const
{
  return Spectrum::black();
} // end ScatteringDistributionFunction::evaluate()

void *ScatteringDistributionFunction
  ::operator new(size_t size, FunctionAllocator &alloc)
{
  return alloc.malloc();
} // end ScatteringDistributionFunction::operator new()

float ScatteringDistributionFunction
  ::evaluatePdf(const Vector &wo,
                const DifferentialGeometry &dg,
                const Vector &wi) const
{
  return Mappings<Vector>::evaluateCosineHemispherePdf(wi, dg.getNormal());
} // end ScatteringDistributionFunction::evaluatePdf()

float ScatteringDistributionFunction
  ::evaluatePdf(const Vector &wo,
                const DifferentialGeometry &dg,
                const Vector &wi,
                const bool delta,
                const ComponentIndex component) const
{
  // just ignore delta & component and evaluate the other method
  return evaluatePdf(wo,dg,wi);
} // end ScatteringDistributionFunction::evaluatePdf()

Spectrum ScatteringDistributionFunction
  ::evaluate(const Vector &wo,
             const DifferentialGeometry &dg,
             const Vector &wi,
             const bool delta,
             const ComponentIndex component,
             float &pdf) const
{
  if(isSpecular()) std::cerr << "ScatteringDistributionFunction::evaluate(): Implement me in the delta function subclass!" << std::endl;
  pdf = evaluatePdf(wo,dg,wi,delta,component);
  return evaluate(wo,dg,wi);
} // end ScatteringDistributionFunction::evaluate()

Spectrum ScatteringDistributionFunction
  ::sample(const Vector &wo,
           const DifferentialGeometry &dg,
           const float u0,
           const float u1,
           const float u2,
           Vector &wi,
           float &pdf,
           bool &delta,
           ComponentIndex &component) const
{
  delta = false;
  component = 0;
  Mappings<Vector>::unitSquareToCosineHemisphere(u0, u1, dg.getTangent(), dg.getBinormal(), dg.getNormal(), wi, pdf);
  return evaluate(wo, dg, wi);
} // end ScatteringDistributionFunction::sample()

Spectrum ScatteringDistributionFunction
  ::sample(const DifferentialGeometry &dg,
           const float u0,
           const float u1,
           const float u2,
           Vector &w,
           float &pdf,
           bool &delta) const
{
  delta = false;
  Mappings<Vector>::unitSquareToCosineHemisphere(u0, u1, dg.getTangent(), dg.getBinormal(), dg.getNormal(), w, pdf);
  return evaluate(w, dg);
} // end ScatteringDistributionFunction::sample()

void ScatteringDistributionFunction
  ::invert(const Vector &w,
           const DifferentialGeometry &dg,
           float &u0,
           float &u1) const
{
  Mappings<Vector>::cosineHemisphereToUnitSquare(w, dg.getTangent(), dg.getBinormal(), dg.getNormal(), u0, u1);
} // end ScatteringDistributionFunction::invert()

float ScatteringDistributionFunction
  ::evaluatePdf(const Vector &w,
                const DifferentialGeometry &dg) const
{
  return Mappings<Vector>::evaluateCosineHemispherePdf(w, dg.getNormal());
} // end ScatteringDistributionFunctionr::evaluatePdf()

bool ScatteringDistributionFunction
  ::isSpecular(void) const
{
  return false;
} // end ScatteringDistributionFunction::isSpecular()

ScatteringDistributionFunction *ScatteringDistributionFunction
  ::clone(FunctionAllocator &allocator) const
{
  ScatteringDistributionFunction *result = static_cast<ScatteringDistributionFunction*>(allocator.malloc());
  if(result != 0)
  {
    // the default implementation does a dumb copy
    memcpy(result, this, sizeof(FunctionAllocator::Block));
  } // end if

  return result;
} // end ScatteringDistributionFunction::clone()

float ScatteringDistributionFunction
  ::evaluateGeometricTerm(const float nDotWo,
                          const float nDotWi,
                          const float nDotWh,
                          const float woDotWh)
{
  float invWoDotWh = 1.0f / woDotWh;
  return std::min(1.0f, std::min(2.0f * nDotWh * nDotWo * invWoDotWh, 2.0f * nDotWh * nDotWi * invWoDotWh));
} // end ScatteringDistributionFunction::evaluateGeometricTerm()

