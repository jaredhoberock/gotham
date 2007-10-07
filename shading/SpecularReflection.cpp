/*! \file SpecularReflection.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of SpecularReflection class.
 */

#include "SpecularReflection.h"
#include "Fresnel.h"

SpecularReflection
  ::SpecularReflection(const Spectrum &r,
                       const float eta)
    :Parent(),mReflectance(r)
{
  mFresnel = new FresnelConductor(mReflectance, Spectrum(eta,eta,eta));
} // end SpecularReflection::SpecularReflection()

SpecularReflection
  ::SpecularReflection(const Spectrum &r,
                       const float etai,
                       const float etat)
    :Parent(),mReflectance(r)
{
  mFresnel = new FresnelDielectric(etai, etat);
} // end SpecularReflection::SpecularReflection()

Spectrum SpecularReflection
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
  delta = true;
  component = 0;

  wi = dg.getNormal().reflect(wo);
  wi = wi.normalize();
  pdf = 1.0f;

  float cosi = dg.getNormal().absDot(wo);

  Spectrum result = mReflectance * mFresnel->evaluate(cosi);

  // must divide by the dot product because
  // we will multiply it later
  return result / dg.getNormal().absDot(wi);
} // end SpecularReflection::sample()

Spectrum SpecularReflection
  ::evaluate(const Vector &wo,
             const DifferentialGeometry &dg,
             const Vector &wi) const
{
  return Spectrum::black();
} // end SpecularReflection::evaluate()

