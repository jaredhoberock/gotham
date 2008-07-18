/*! \file PhongReflectionBase.inl
 *  \author Jared Hoberock
 *  \brief Inline file for PhongReflectionBase.h.
 */

#include "PhongReflectionBase.h"
#include "../areSameHemisphere.h"
#include "../../geometry/Mappings.h"
#include "fresnel.h"

#ifndef INV_TWOPI
#define INV_TWOPI (1.0f / (2.0f * M_PI))
#endif // INV_TWOPI

template<typename V3, typename S3, typename Boolean>
  PhongReflectionBase<V3,S3,Boolean>
    ::PhongReflectionBase(const Spectrum &r,
                          const float eta,
                          const float exponent)
      :mReflectance(r),mEtai(0),mEtat(eta),
       mExponent(exponent),mAbsorptionCoefficient(approximateFresnelAbsorptionCoefficient(r))
{
  ;
} // end PhongReflectionBase::PhongReflectionBase()

template<typename V3, typename S3, typename Boolean>
  PhongReflectionBase<V3,S3,Boolean>
    ::PhongReflectionBase(const Spectrum &r,
                          const float etai,
                          const float etat,
                          const float exponent)
      :mReflectance(r),mEtai(etai),mEtat(etat),mExponent(exponent)
{
  ;
} // end PhongReflectionBase::PhongReflectionBase()

template<typename V3, typename S3, typename Boolean>
  S3 PhongReflectionBase<V3,S3,Boolean>
    ::evaluate(const Vector &wo,
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
  Vector m = normalize(wo + wi);

  // evaluate the phong distribution
  // XXX hide the phong distribution evaluate somewhere else
  float D = (mExponent + 2.0f) * powf(dot(normal,m), mExponent) * INV_TWOPI;

  // note we can use either wo or wi here, since they are reflections about m
  float cosi = fabs(dot(m,wo));

  // compute the geometry term (or not)
  float G = 1.0f;

  // evaluate Fresnel
  Spectrum F;
  if(mEtai == 0)
  {
    // evaluate conductor
    F = evaluateFresnelConductor(mAbsorptionCoefficient, mEtat, cosi);
  } // end if
  else
  {
    // evaluate dielectric
    F.x = evaluateFresnelDielectricUnknownOrientation(mEtai,mEtat,cosi);
    F.y = F.x;
    F.z = F.x;
  } // end else

  // Walter et al, 2007, equation 20
  result = mReflectance * F * G * D;
  result /= (4.0f * fabs(dot(normal,wo) * dot(normal,wi)));

  return result;
} // end PhongReflectionBase::evaluate()

template<typename V3, typename S3, typename Boolean>
  S3 PhongReflectionBase<V3,S3,Boolean>
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
} // end PhongReflection::sample()

template<typename V3, typename S3, typename Boolean>
  S3 PhongReflectionBase<V3,S3,Boolean>
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
  Vector m;
  Mappings<Vector>::unitSquareToPhongLobe(u0, u1, normal, mExponent,
                                          tangent, binormal, normal,
                                          m, pdf);

  // we are able to sample the Phong distribution exactly
  float D = pdf;

  // reflect wo about the microfacet normal
  wi = normalize(reflect(m,wo));

  Spectrum result;
  result.x = 0;
  result.y = 0;
  result.z = 0;

  // wo & wi must lie in the same hemisphere
  if(!areSameHemisphere(wi,normal,wo)) return result;

  // note we can use either wo or wi here, since they are reflections about m
  float cosi = fabs(dot(m,wo));

  // compute the geometry term
  float nDotWo = dot(normal,wo);
  float nDotWi = dot(normal,wi);
  float G = 1.0f;

  // Walter et al, 2007, equation 14
  // XXX use cosi in the denominator here
  float J = 0.25f / fabs(dot(m,wi));
  pdf *= J;

  // evaluate Fresnel
  Spectrum F;
  if(mEtai == 0)
  {
    // evaluate conductor
    F = evaluateFresnelConductor(mAbsorptionCoefficient, mEtat, cosi);
  } // end if
  else
  {
    // evaluate dielectric
    F.x = evaluateFresnelDielectricUnknownOrientation(mEtai,mEtat,cosi);
    F.y = F.x;
    F.z = F.x;
  } // end else

  // Walter et al, 2007, equation 20
  result = mReflectance * F * G * D;
  result /= (4.0f * fabs(nDotWo * nDotWi));

  return result;
} // end PhongReflection::sample()

template<typename V3, typename S3, typename Boolean>
  float PhongReflectionBase<V3,S3,Boolean>
    ::evaluatePdf(const Vector &wo,
                  const Vector &point,
                  const Vector &tangent,
                  const Vector &binormal,
                  const Vector &normal,
                  const Vector &wi) const
{
  return evaluatePdf(wo,tangent,binormal,normal,wi);
} // end PhongReflectionBase::evaluatePdf()

template<typename V3, typename S3, typename Boolean>
  float PhongReflectionBase<V3,S3,Boolean>
    ::evaluatePdf(const Vector &wo,
                  const Vector &tangent,
                  const Vector &binormal,
                  const Vector &normal,
                  const Vector &wi) const
{
  // wo & wi must lie in the same hemisphere
  if(!areSameHemisphere(wi,normal,wo)) return 0;

  // compute the microfacet normal (half-vector)
  Vector m = normalize(wo + wi);

  // Walter et al, 2007, equation 14
  float J = 0.25f / fabs(dot(m,wi));

  // evaluate the phong distribution and multiply by the Jacobian
  // XXX hide the phong distribution evaluation somewhere else
  return (mExponent + 2.0f) * powf(fabs(dot(normal,m)), mExponent) * INV_TWOPI * J;
} // end PhongReflectionBase::evaluatePdf()

