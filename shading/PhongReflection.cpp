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
                    const float exponent,
                    FunctionAllocator &alloc)
    :mReflectance(r),mExponent(exponent)
{
  mFresnel = new(alloc) FresnelConductor(mReflectance, Spectrum(eta,eta,eta));
} // end PhongReflection::PhongReflection()

PhongReflection
  ::PhongReflection(const Spectrum &r,
                    const float etai,
                    const float etat,
                    const float exponent,
                    FunctionAllocator &alloc)
    :mReflectance(r),mExponent(exponent)
{
  mFresnel = new(alloc) FresnelDielectric(etai, etat);
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
  delta = false;
  index = 0;

  // sample a microfacet normal
  Vector m;
  Mappings::unitSquareToPhongLobe(u0, u1, dg.getNormal(), mExponent,
                                  dg.getTangent(), dg.getBinormal(), dg.getNormal(),
                                  m, pdf);

  // we are able to sample the Phong distribution exactly
  float D = pdf;

  // reflect wo about the microfacet normal
  wi = m.reflect(wo).normalize();

  // wo & wi must lie in the same hemisphere
  if(!areSameHemisphere(wi,dg.getNormal(),wo)) return Spectrum::black();

  // note we can use either wo or wi here, since they are reflections about m
  float cosi = m.absDot(wo);

  // compute the geometry term
  float nDotWo = dg.getNormal().absDot(wo);
  float nDotWi = dg.getNormal().absDot(wi);
  float G = 1.0f;

  // Walter et al, 2007, equation 14
  float J = 0.25f / m.absDot(wi);
  pdf *= J;

  // Walter et al, 2007, equation 20
  Spectrum result = mReflectance * mFresnel->evaluate(cosi) * G * D;
  result /= (4.0f * nDotWo * nDotWi);

  return result;
} // end PhongReflection::sample()

float PhongReflection
  ::evaluatePdf(const Vector3 &wo,
                const DifferentialGeometry &dg,
                const Vector3 &wi) const
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
  // wo & wi must lie in the same hemisphere
  if(!areSameHemisphere(wi,dg.getNormal(),wo)) return Spectrum::black();

  // compute the microfacet normal (half-vector)
  Vector m = (wo + wi).normalize();

  // evaluate the phong distribution
  // XXX hide the phong distribution evaluation somewhere else
  float D = (mExponent + 2.0f) * powf(dg.getNormal().absDot(m), mExponent) * INV_TWOPI;

  // note we can use either wo or wi here, since they are reflections about m
  float cosi = m.absDot(wo);

  // compute the geometry term
  float G = 1.0f;

  // Walter et al, 2007, equation 20
  Spectrum result = mReflectance * mFresnel->evaluate(cosi) * G * D;
  result /= (4.0f * dg.getNormal().absDot(wo) * dg.getNormal().absDot(wi));

  return result;
} // end PhongReflection::evaluate()

Spectrum PhongReflection
  ::evaluate(const Vector &wo,
             const DifferentialGeometry &dg,
             const Vector &wi,
             const bool delta,
             const ComponentIndex component,
             float &pdf) const
{
  // wo & wi must lie in the same hemisphere
  pdf = 0;
  if(!areSameHemisphere(wi,dg.getNormal(),wo)) return Spectrum::black();

  // compute the microfacet normal (half-vector)
  Vector m = (wo + wi).normalize();

  // evaluate the phong distribution
  // XXX hide the phong distribution evaluation somewhere else
  float D = (mExponent + 2.0f) * powf(dg.getNormal().absDot(m), mExponent) * INV_TWOPI;

  // Walter et al, 2007, equation 14
  float J = 0.25f / m.absDot(wi);
  pdf = D * J;

  // note we can use either wo or wi here, since they are reflections about m
  float cosi = m.absDot(wo);

  // compute the geometry term
  float G = 1.0f;

  // Walter et al, 2007, equation 20
  Spectrum result = mReflectance * mFresnel->evaluate(cosi) * G * D;
  result /= (4.0f * dg.getNormal().absDot(wo) * dg.getNormal().absDot(wi));

  return result;
} // end PhongReflection::evaluate()

ScatteringDistributionFunction *PhongReflection
  ::clone(FunctionAllocator &allocator) const
{
  PhongReflection *result = static_cast<PhongReflection*>(Parent::clone(allocator));
  if(result != 0)
  {
    // clone the Fresnel
    result->mFresnel = static_cast<Fresnel*>(allocator.malloc());
    if(result->mFresnel != 0)
    {
      memcpy(result->mFresnel, mFresnel, sizeof(FunctionAllocator::Block));
    } // end if
    else
    {
      // failure
      result = 0;
    } // end else
  } // end if

  return result;
} // end PhongReflection::clone()

