/*! \file AshikhminShirleyReflection.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of AshikhminShirleyReflection class.
 */

#include "AshikhminShirleyReflection.h"
#include "Fresnel.h"
#include "../geometry/Mappings.h"

AshikhminShirleyReflection
  ::AshikhminShirleyReflection(const Spectrum &r,
                    const float eta,
                    const float uExponent,
                    const float vExponent)
    :mReflectance(r),mNu(uExponent),mNv(vExponent)
{
  mFresnel = new FresnelConductor(mReflectance, Spectrum(eta,eta,eta));
} // end AshikhminShirleyReflection::AshikhminShirleyReflection()

Spectrum AshikhminShirleyReflection
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
  Vector wh;
  Mappings::unitSquareToAnisotropicPhongLobe(u0, u1,
                                             mNu, mNv,
                                             dg.getTangent(), dg.getBinormal(), dg.getNormal(),
                                             wh, pdf);
  // compute cos theta wh
  float cosThetaH = wh.absDot(wi);

  // we are able to sample the Ashikhmin-Shirley distribution exactly
  float D = pdf;

  // reflect wo about the microfacet normal
  wi = wh.reflect(wo).normalize();

  // wo & wi must lie in the same hemisphere
  if(!areSameHemisphere(wi,dg.getNormal(),wo)) return Spectrum::black();

  // compute the geometry term
  float nDotWo = dg.getNormal().absDot(wo);
  float nDotWi = dg.getNormal().absDot(wi);
  float nDotWh = dg.getNormal().absDot(wh);
  float woDotWh = wh.absDot(wo);
  //float G = evaluateGeometricTerm(nDotWo, nDotWi, nDotWh, woDotWh);
  float G = 1.0f;

  pdf *= (0.25f / woDotWh);

  // Walter et al, 2007, equation 20
  // note we can use either wi or wo here because they are reflections about wh
  float whDotWi = wh.dot(wi);
  Spectrum result = mReflectance * mFresnel->evaluate(whDotWi) * G * D;
  result /= (4.0f * nDotWo * nDotWi);

  return result;
} // end AshikhminShirleyReflection::sample()

Spectrum AshikhminShirleyReflection
  ::evaluate(const Vector &wo,
             const DifferentialGeometry &dg,
             const Vector &wi) const
{
  // wo & wi must lie in the same hemisphere
  if(!areSameHemisphere(wi,dg.getNormal(),wo)) return Spectrum::black();

  // compute the microfacet normal (half-vector)
  Vector wh = (wo + wi).normalize();

  // compute cos theta wh
  // note we can use either wi or wo here because they are reflections about wh
  float cosThetaH = wh.absDot(wi);

  // evaluate the Ashikhmin-Shirley distribution
  float D = Mappings::evaluateAnisotropicPhongLobePdf(wh, mNu, mNv,
                                                      dg.getTangent(),
                                                      dg.getBinormal(),
                                                      dg.getNormal());

  // compute the geometry term
  float nDotWo = dg.getNormal().absDot(wo);
  float nDotWi = dg.getNormal().absDot(wi);
  float nDotWh = dg.getNormal().absDot(wh);
  float woDotWh = wh.absDot(wo);
  //float G = evaluateGeometricTerm(nDotWo, nDotWi, nDotWh, woDotWh);
  float G = 1.0f;

  // Walter et al, 2007, equation 20
  Spectrum result = mReflectance * mFresnel->evaluate(cosThetaH) * G * D;
  result /= (4.0f * nDotWo * nDotWi);

  return result;
} // end AshikhminShirleyReflection::evaluate()

float AshikhminShirleyReflection
  ::evaluatePdf(const Vector3 &wo,
                const DifferentialGeometry &dg,
                const Vector3 &wi) const
{
  // wo & wi must lie in the same hemisphere
  if(!areSameHemisphere(wi,dg.getNormal(),wo)) return 0;

  // compute the microfacet normal (half-vector)
  Vector wh = (wo + wi).normalize();

  // Walter et al, 2007, equation 14
  float J = 0.25f / wh.absDot(wi);

  // evaluate the AshikhminShirley distribution and multiply by the Jacobian
  return J * Mappings::evaluateAnisotropicPhongLobePdf(wh, mNu, mNv,
                                                       dg.getTangent(), dg.getBinormal(), dg.getNormal());
} // end AshikhminShirleyReflection::evaluatePdf()

ScatteringDistributionFunction *AshikhminShirleyReflection
  ::clone(FunctionAllocator &allocator) const
{
  AshikhminShirleyReflection *result = static_cast<AshikhminShirleyReflection*>(Parent::clone(allocator));
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
} // end AshikhminShirleyReflection::clone()

