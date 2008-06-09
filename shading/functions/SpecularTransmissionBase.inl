/*! \file SpecularTransmissionBase.inl
 *  \author Jared Hoberock
 *  \brief Inline file for SpecularTransmissionBase.h.
 */

#include "SpecularTransmissionBase.h"
#include "fresnel.h"

template<typename V3, typename S3>
  SpecularTransmissionBase<V3,S3>
    ::SpecularTransmissionBase(const Spectrum &t,
                               const float etai,
                               const float etat)
      :mTransmittance(t),mEtai(etai),mEtat(etat)
{
  ;
} // end SpecularTransmissionBase::SpecularTransmissionBase()

template<typename V3, typename S3>
  S3 SpecularTransmissionBase<V3,S3>
    ::evaluate(void) const
{
  Spectrum result;
  result.x = 0;
  result.y = 0;
  result.z = 0;
  return result;
} // end SpecularTransmissionBase::evaluate()

template<typename V3, typename S3>
  S3 SpecularTransmissionBase<V3,S3>
    ::evaluate(const Vector &wo,
               const Vector &normal,
               const Vector &wi) const
{
  return evaluate();
} // end SpecularTransmissionBase::evaluate()

template<typename V3, typename S3>
  S3 SpecularTransmissionBase<V3,S3>
    ::sample(const Vector &wo,
             const Vector &normal,
             Vector &wi,
             float &pdf,
             bool &delta,
             unsigned int &component) const
{
  delta = true;
  component = 0;
  Spectrum result;
  result.x = 0;
  result.y = 0;
  result.z = 0;

  pdf = 1.0f;

  float cosi = dot(wo,normal);
  bool entering = cosi > 0;
  float ei = mEtai, et = mEtat;
  if(!entering) 
  {
    float temp = ei;
    ei = et;
    et = temp;
  } // end if

  // compute refracted ray direction
  float sini2 = 1.0f - cosi*cosi;
  float eta = ei / et;
  float sint2 = eta * eta * sini2;

  // check for total internal refraction
  if(sint2 <= 1.0f)
  {
    float cost = 1.0f - sint2;

    // clamp cost to 0
    cost = cost < 0 ? 0 : cost;

    cost = -sqrtf(cost);
    if(entering) cost = -cost;

    wi = (eta*cosi - cost)*normal - eta * wo;

    Spectrum f;
    f.x = 1;
    f.y = 1;
    f.z = 1;

    // compute fresnel term
    float fresnel = evaluateFresnelDielectric(ei,et,cosi,cost);
    f.x -= fresnel;
    f.y -= fresnel;
    f.z -= fresnel;

    f = saturate(f);

    result = (et*et)/(ei*ei) * f * mTransmittance;

    result /= fabs(dot(normal,wi));
  } // end if

  return result;
} // end SpecularTransmissionBase::sample()

template<typename V3, typename S3>
  S3 SpecularTransmissionBase<V3,S3>
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
} // end SpecularTransmissionBase::sample()

template<typename V3, typename S3>
  float SpecularTransmissionBase<V3,S3>
    ::evaluatePdf(const Vector &wo,
                  const Vector &point,
                  const Vector &tangent,
                  const Vector &binormal,
                  const Vector &normal,
                  const Vector &wi) const
{
  return 0;
} // end SpecularTransmissionBase::evaluatePdf()

