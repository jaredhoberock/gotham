/*! \file AshikhminShirleyReflection.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of AshikhminShirleyReflection class.
 */

#include "AshikhminShirleyReflection.h"
#include "../geometry/Mappings.h"

AshikhminShirleyReflection
  ::AshikhminShirleyReflection(const Spectrum &r,
                               const float eta,
                               const float uExponent,
                               const float vExponent)
    :mReflectance(r),mNu(uExponent),mNv(vExponent),mFresnel(mReflectance, Spectrum(eta,eta,eta))
{
  ;
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
  Mappings<Vector>::unitSquareToAnisotropicPhongLobe(u0, u1,
                                                     mNu, mNv,
                                                     dg.getTangent(), dg.getBinormal(), dg.getNormal(),
                                                     wh, pdf);

  // we are able to sample the Ashikhmin-Shirley distribution exactly
  float D = pdf;

  // reflect wo about the microfacet normal
  wi = wh.reflect(wo).normalize();

  // wo & wi must lie in the same hemisphere
  if(!areSameHemisphere(wi,dg.getNormal(),wo)) return Spectrum::black();

  // compute the geometry term
  float nDotWo = dg.getNormal().absDot(wo);
  float nDotWi = dg.getNormal().absDot(wi);
  float woDotWh = wh.absDot(wo);
  //float nDotWh = dg.getNormal().absDot(wh);
  //float G = evaluateGeometricTerm(nDotWo, nDotWi, nDotWh, woDotWh);
  float G = 1.0f;

  pdf *= (0.25f / woDotWh);

  // Walter et al, 2007, equation 20
  // note we can use either wi or wo here because they are reflections about wh
  float whDotWi = wh.dot(wi);
  Spectrum result = mReflectance * mFresnel.evaluate(whDotWi) * G * D;
  result /= (4.0f * nDotWo * nDotWi);

  return result;
} // end AshikhminShirleyReflection::sample()

Spectrum AshikhminShirleyReflection
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
  Vector wh = (wo + wi).normalize();

  // compute cos theta wh
  // note we can use either wi or wo here because they are reflections about wh
  float cosThetaH = wh.absDot(wi);

  // evaluate the Ashikhmin-Shirley distribution
  float D = Mappings<Vector>::evaluateAnisotropicPhongLobePdf(wh, mNu, mNv,
                                                              dg.getTangent(),
                                                              dg.getBinormal(),
                                                              dg.getNormal());

  // Walter et al, 2007, equation 14
  float J = 0.25f / wh.absDot(wi);
  pdf = D * J;

  // compute the geometry term
  float nDotWo = dg.getNormal().absDot(wo);
  float nDotWi = dg.getNormal().absDot(wi);
  //float woDotWh = wh.absDot(wo);
  //float nDotWh = dg.getNormal().absDot(wh);
  //float G = evaluateGeometricTerm(nDotWo, nDotWi, nDotWh, woDotWh);
  float G = 1.0f;

  // Walter et al, 2007, equation 20
  Spectrum result = mReflectance * mFresnel.evaluate(cosThetaH) * G * D;
  result /= (4.0f * nDotWo * nDotWi);

  return result;
} // end AshikhminShirleyReflection::evaluate()

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
  float D = Mappings<Vector>::evaluateAnisotropicPhongLobePdf(wh, mNu, mNv,
                                                              dg.getTangent(),
                                                              dg.getBinormal(),
                                                              dg.getNormal());

  // compute the geometry term
  float nDotWo = dg.getNormal().absDot(wo);
  float nDotWi = dg.getNormal().absDot(wi);
  //float woDotWh = wh.absDot(wo);
  //float nDotWh = dg.getNormal().absDot(wh);
  //float G = evaluateGeometricTerm(nDotWo, nDotWi, nDotWh, woDotWh);
  float G = 1.0f;

  // Walter et al, 2007, equation 20
  Spectrum result = mReflectance * mFresnel.evaluate(cosThetaH) * G * D;
  result /= (4.0f * nDotWo * nDotWi);

  return result;
} // end AshikhminShirleyReflection::evaluate()

float AshikhminShirleyReflection
  ::evaluatePdf(const Vector &wo,
                const DifferentialGeometry &dg,
                const Vector &wi) const
{
  // wo & wi must lie in the same hemisphere
  if(!areSameHemisphere(wi,dg.getNormal(),wo)) return 0;

  // compute the microfacet normal (half-vector)
  Vector wh = (wo + wi).normalize();

  // Walter et al, 2007, equation 14
  float J = 0.25f / wh.absDot(wi);

  // evaluate the AshikhminShirley distribution and multiply by the Jacobian
  return J * Mappings<Vector>::evaluateAnisotropicPhongLobePdf(wh, mNu, mNv,
                                                               dg.getTangent(), dg.getBinormal(), dg.getNormal());
} // end AshikhminShirleyReflection::evaluatePdf()

