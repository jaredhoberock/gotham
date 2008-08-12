/*! \file AshikhminShirleyReflectionBase.inl
 *  \author Jared Hoberock
 *  \brief Inline file for AshikhminShirleyReflectionBase.h.
 */

#include "AshikhminShirleyReflectionBase.h"
#include "../areSameHemisphere.h"
#include "../../geometry/Mappings.h"
#include "fresnel.h"

template<typename V3, typename S3, typename Boolean>
  AshikhminShirleyReflectionBase<V3,S3,Boolean>
    ::AshikhminShirleyReflectionBase(const Spectrum &r,
                                     const float eta,
                                     const float uExponent,
                                     const float vExponent)
      :mReflectance(r),
       mEtai(0),
       mEtat(eta),
       mNu(uExponent),
       mNv(vExponent),
       mAbsorptionCoefficient(approximateFresnelAbsorptionCoefficient(r))
{
  ;
} // end AshikhminShirleyReflectionBase::AshikhminShirleyReflectionBase()

template<typename V3, typename S3, typename Boolean>
  AshikhminShirleyReflectionBase<V3,S3,Boolean>
    ::AshikhminShirleyReflectionBase(const Spectrum &r,
                                     const float etai,
                                     const float etat,
                                     const float uExponent,
                                     const float vExponent)
      :mReflectance(r),
       mEtai(etai),
       mEtat(etat),
       mNu(uExponent),
       mNv(vExponent)
{
  ;
} // end AshikhminShirleyReflectionBase::AshikhminShirleyReflectionBase()

template<typename V3, typename S3, typename Boolean>
  S3 AshikhminShirleyReflectionBase<V3,S3,Boolean>
    ::evaluate(const Vector &wo,
               const Vector &tangent,
               const Vector &binormal,
               const Vector &normal,
               const Vector &wi) const
{
  Spectrum result;
  result.x = 0;
  result.y = 0;
  result.z = 0;

  // wo & wi must lie in the same hemisphere
  if(!areSameHemisphere(wi,normal,wo)) return result;

  // compute the microfacet normal (half-vector)
  Vector wh = normalize(wo + wi);

  // compute cos theta wh
  // note we can use either wi or wo here because they are reflections about wh
  float cosThetaH = fabs(dot(wh,wi));

  // evaluate the Ashikhmin-Shirley distribution
  float D = Mappings<Vector>::evaluateAnisotropicPhongLobePdf(wh, mNu, mNv,
                                                              tangent,
                                                              binormal,
                                                              normal);

  // compute the geometry term
  float nDotWo = fabs(dot(normal,wo));
  float nDotWi = fabs(dot(normal,wi));
  //float woDotWh = wh.absDot(wo);
  //float nDotWh = dg.getNormal().absDot(wh);
  //float G = evaluateGeometricTerm(nDotWo, nDotWi, nDotWh, woDotWh);
  float G = 1.0f;

  // evaluate Fresnel
  Spectrum F;
  if(mEtai == 0)
  {
    // evaluate conductor
    F = evaluateFresnelConductor(mAbsorptionCoefficient, mEtat, nDotWi);
  } // end if
  else
  {
    // evaluate dielectric
    F.x = evaluateFresnelDielectricUnknownOrientation(mEtai,mEtat,nDotWi);
    F.y = F.x;
    F.z = F.x;
  } // end else

  // Walter et al, 2007, equation 20
  result = mReflectance * F * G * D;
  result /= (4.0f * nDotWo * nDotWi);

  return result;
} // end AshihkminShirleyReflectionBase::evaluate()

template<typename V3, typename S3, typename Boolean>
  S3 AshikhminShirleyReflectionBase<V3,S3,Boolean>
    ::sample(const Vector &wo,
             const Vector &point,
             const Vector &tangent,
             const Vector &binormal,
             const Vector &normal,
             const float u0,
             const float u1,
             const float u2,
             Vector &wi,
             float &pdf,
             Boolean &delta,
             unsigned int &index) const
{
  return sample(wo,tangent,binormal,normal,u0,u1,u2,wi,pdf,delta,index);
} // end AshikhminShirleyReflectionBase::sample()

template<typename V3, typename S3, typename Boolean>
  S3 AshikhminShirleyReflectionBase<V3,S3,Boolean>
    ::sample(const Vector &wo,
             const Vector &tangent,
             const Vector &binormal,
             const Vector &normal,
             const float u0,
             const float u1,
             const float u2,
             Vector &wi,
             float &pdf,
             Boolean &delta,
             unsigned int &index) const
{
  delta = false;
  index = 0;

  // sample a microfacet normal
  Vector wh;
  Mappings<Vector>::unitSquareToAnisotropicPhongLobe(u0, u1,
                                                     mNu, mNv,
                                                     tangent, binormal, normal,
                                                     wh, pdf);

  // we are able to sample the Ashikhmin-Shirley distribution exactly
  float D = pdf;

  // reflect wo about the microfacet normal
  wi = normalize(reflect(wh,wo));

  Spectrum result;
  result.x = 0;
  result.y = 0;
  result.z = 0;

  // wo & wi must lie in the same hemisphere
  if(!areSameHemisphere(wi,normal,wo)) return result;

  // compute the geometry term
  float nDotWo = fabs(dot(normal,wo));
  float nDotWi = fabs(dot(normal,wi));
  float woDotWh = fabs(dot(wh,wo));
  //float nDotWh = dg.getNormal().absDot(wh);
  //float G = evaluateGeometricTerm(nDotWo, nDotWi, nDotWh, woDotWh);
  float G = 1.0f;

  pdf *= (0.25f / woDotWh);

  // evaluate Fresnel
  Spectrum F;
  if(mEtai == 0)
  {
    // evaluate conductor
    F = evaluateFresnelConductor(mAbsorptionCoefficient, mEtat, nDotWi);
  } // end if
  else
  {
    // evaluate dielectric
    F.x = evaluateFresnelDielectricUnknownOrientation(mEtai,mEtat,nDotWi);
    F.y = F.x;
    F.z = F.x;
  } // end else

  // Walter et al, 2007, equation 20
  result = mReflectance * F * G * D;
  result /= (4.0f * nDotWo * nDotWi);

  return result;
} // end AshikhminShirleyReflectionBase::sample()

template<typename V3, typename S3, typename Boolean>
  float AshikhminShirleyReflectionBase<V3,S3,Boolean>
    ::evaluatePdf(const Vector &wo,
                  const Vector &point,
                  const Vector &tangent,
                  const Vector &binormal,
                  const Vector &normal,
                  const Vector &wi) const
{
  return evaluatePdf(wo,tangent,binormal,normal,wi);
} // end AshikhminShirleyReflectionBase::evaluatePdf()

template<typename V3, typename S3, typename Boolean>
  float AshikhminShirleyReflectionBase<V3,S3,Boolean>
    ::evaluatePdf(const Vector &wo,
                  const Vector &tangent,
                  const Vector &binormal,
                  const Vector &normal,
                  const Vector &wi) const
{
  // wo & wi must lie in the same hemisphere
  if(!areSameHemisphere(wi,normal,wo)) return 0;

  // compute the microfacet normal (half-vector)
  Vector wh = normalize(wo + wi);

  // Walter et al, 2007, equation 14
  float J = 0.25f / fabs(dot(wh,wi));

  // evaluate the AshikhminShirley distribution and multiply by the Jacobian
  return J * Mappings<Vector>::evaluateAnisotropicPhongLobePdf(wh, mNu, mNv,
                                                               tangent, binormal, normal);
} // end AshikhminShirleyReflectionBase::evaluatePdf()

