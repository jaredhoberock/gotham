/*! \file PhongReflection.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of PhongReflection class.
 */

#include "PhongReflection.h"
#include "Fresnel.h"
#include "../geometry/Mappings.h"

PhongReflection
  ::PhongReflection(const Spectrum &r,
                    const float eta,
                    const float exponent)
    :Parent1(r,eta,exponent)
{
  ;
} // end PhongReflection::PhongReflection()

PhongReflection
  ::PhongReflection(const Spectrum &r,
                    const float etai,
                    const float etat,
                    const float exponent)
    :Parent1(r,etai,etat,exponent)
{
  ;
} // end PhongReflection::PhongReflection()

Spectrum PhongReflection
  ::sample(const Vector &wo,
           const DifferentialGeometry &dg,
           const float u0,
           const float u1,
           const float u2,
           Vector &wi,
           float &pdf,
           bool &delta,
           ComponentIndex &index) const
{
  return Parent1::sample(wo,dg.getTangent(),dg.getBinormal(),dg.getNormal(),
                         u0,u1,u2,wi,pdf,delta,index);
} // end PhongReflection::sample()

float PhongReflection
  ::evaluatePdf(const Vector &wo,
                const DifferentialGeometry &dg,
                const Vector &wi) const
{
  // wo & wi must lie in the same hemisphere
  if(!areSameHemisphere(wi,dg.getNormal(),wo)) return 0;

  // compute the microfacet normal (half-vector)
  Vector m = (wo + wi).normalize();

  // Walter et al, 2007, equation 14
  float J = 0.25f / m.absDot(wi);

  // evaluate the phong distribution and multiply by the Jacobian
  // XXX hide the phong distribution evaluation somewhere else
  return (mExponent + 2.0f) * powf(dg.getNormal().absDot(m), mExponent) * INV_TWOPI * J;
} // end PhongReflection::evaluatePdf()

Spectrum PhongReflection
  ::evaluate(const Vector &wo,
             const DifferentialGeometry &dg,
             const Vector &wi) const
{
  return Parent1::evaluate(wo,dg.getNormal(),wi);
} // end PhongReflection::evaluate()

Spectrum PhongReflection
  ::evaluate(const Vector &wo,
             const DifferentialGeometry &dg,
             const Vector &wi,
             const bool delta,
             const ComponentIndex component,
             float &pdf) const
{
  return Parent1::evaluate(wo,dg.getNormal(),wi);
  pdf = evaluatePdf(wo,dg,wi);
} // end PhongReflection::evaluate()

