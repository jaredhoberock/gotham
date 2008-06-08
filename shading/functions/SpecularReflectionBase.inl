/*! \file SpecularReflectionBase.inl
 *  \author Jared Hoberock
 *  \brief Inline file for SpecularReflectionBase.h.
 */

#include "SpecularReflectionBase.h"
#include "fresnel.h"

template<typename V3, typename S3>
  SpecularReflectionBase<V3,S3>
    ::SpecularReflectionBase(const Spectrum &r,
                             const float eta)
      :mReflectance(r),mEtai(0),mEtat(eta),
       mAbsorptionCoefficient(approximateFresnelAbsorptionCoefficient(r))
{
  ;
} // end PhongReflectionBase::PhongReflectionBase()

template<typename V3, typename S3>
  SpecularReflectionBase<V3,S3>
    ::SpecularReflectionBase(const Spectrum &r,
                             const float etai,
                             const float etat)
      :mReflectance(r),mEtai(etai),mEtat(etat)
{
  ;
} // end PhongReflectionBase::PhongReflectionBase()

template<typename V3, typename S3>
  S3 SpecularReflectionBase<V3,S3>
    ::evaluate(void) const
{
  S3 result;
  result.x = 0;
  result.y = 0;
  result.z = 0;
  return result;
} // end SpecularReflectionBase::evaluate()

template<typename V3, typename S3>
  S3 SpecularReflectionBase<V3,S3>
    ::evaluate(const Vector &wo,
               const Vector &normal,
               const Vector &wi) const
{
  return evaluate();
} // end SpecularReflectionBase::evaluate()

template<typename V3, typename S3>
  S3 SpecularReflectionBase<V3,S3>
    ::sample(const Vector &wo,
             const Vector &normal,
             Vector &wi,
             float &pdf,
             bool &delta,
             unsigned int &component) const
{
  delta = true;
  component = 0;

  // reflect no doubt needs to compute cosi
  // pass this as an argument to avoid that computation
  wi = reflect(normal,wo);

  // if normal and wo are unit-length, no need to normalize this
  //wi = normalize(wi);
  
  pdf = 1.0f;

  float cosi = dot(normal,wo);

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

  Spectrum result = mReflectance * F;

  // must divide by the dot product because
  // we will multiply it later
  // BUG #1
  return result / fabs(dot(normal,wi));
} // end SpecularReflectionBase::sample()

template<typename V3, typename S3>
  S3 SpecularReflectionBase<V3,S3>
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
             bool &delta,
             unsigned int &component) const
{
  return sample(wo,normal,wi,pdf,delta,component);
} // end SpecularReflectionBase::sample()

template<typename V3, typename S3>
  float SpecularReflectionBase<V3,S3>
    ::evaluatePdf(const Vector &wo,
                  const Vector &point,
                  const Vector &tangent,
                  const Vector &binormal,
                  const Vector &normal,
                  const Vector &wi) const
{
  return 0;
} // end SpecularReflectionBase::evaluatePdf()

