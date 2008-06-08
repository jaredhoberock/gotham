/*! \file LambertianBase.inl
 *  \author Jared Hoberock
 *  \brief Inline file for LambertianBase.h.
 */

#include "LambertianBase.h"
#include "../areSameHemisphere.h"
#include "../../geometry/Mappings.h"

#ifndef INV_PI
#define INV_PI 0.318309886f
#endif // INV_PI

template<typename V3, typename S3>
  LambertianBase<V3,S3>
    ::LambertianBase(const Spectrum &albedo)
      :mAlbedo(albedo)
{
  mAlbedoOverPi = albedo;

  mAlbedoOverPi.x *= INV_PI;
  mAlbedoOverPi.y *= INV_PI;
  mAlbedoOverPi.z *= INV_PI;
} // end LambertianBase::LambertianBase()

template<typename V3, typename S3>
  S3 LambertianBase<V3,S3>
    ::evaluate(const V3 &wo,
               const V3 &normal,
               const V3 &wi) const
{
  S3 result;
  result.x = 0;
  result.y = 0;
  result.z = 0;

  if(areSameHemisphere(wi, normal, wo))
  {
    result = mAlbedoOverPi;
  } // end if

  return result;
} // end LambertianBase::evaluate()

template<typename V3, typename S3>
  const S3 &LambertianBase<V3,S3>
    ::getAlbedo(void) const
{
  return mAlbedo;
} // end LambertianBase::getAlbedo()

template<typename V3, typename S3>
  S3 LambertianBase<V3,S3>
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
  return sample(wo,tangent,binormal,normal,u0,u1,u2,wi,pdf,delta,component);
} // end LambertianBase::sample()

template<typename V3, typename S3>
  S3 LambertianBase<V3,S3>
    ::sample(const Vector &wo,
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
  delta = false;
  component = 0;
  Mappings<V3>::unitSquareToCosineHemisphere(u0, u1, tangent, binormal, normal, wi, pdf);
  return evaluate(wo, normal, wi);
} // end LambertianBase::sample()

template<typename V3, typename S3>
  float LambertianBase<V3,S3>
    ::evaluatePdf(const Vector &wo,
                  const Vector &point,
                  const Vector &tangent,
                  const Vector &binormal,
                  const Vector &normal,
                  const Vector &wi) const
{
  return evaluatePdf(wo,tangent,binormal,normal,wi);
} // end LambertianBase::evaluatePdf()

template<typename V3, typename S3>
  float LambertianBase<V3,S3>
    ::evaluatePdf(const Vector &wo,
                  const Vector &tangent,
                  const Vector &binormal,
                  const Vector &normal,
                  const Vector &wi) const
{
  return Mappings<V3>::evaluateCosineHemispherePdf(wi,normal);
} // end LambertianBase::evaluatePdf()

