/*! \file SpecularTransmissionBase.inl
 *  \author Jared Hoberock
 *  \brief Inline file for SpecularTransmissionBase.h.
 */

#include "SpecularTransmissionBase.h"
#include "fresnel.h"

template<typename V3, typename S3, typename DG>
  SpecularTransmissionBase<V3,S3,DG>
    ::SpecularTransmissionBase(const Spectrum &t,
                               const float etai,
                               const float etat)
      :mTransmittance(t),mEtai(etai),mEtat(etat)
{
  ;
} // end SpecularTransmissionBase::SpecularTransmissionBase()

template<typename V3, typename S3, typename DG>
  S3 SpecularTransmissionBase<V3,S3,DG>
    ::evaluate(void) const
{
  Spectrum result;
  result.x = 0;
  result.y = 0;
  result.z = 0;
  return result;
} // end SpecularTransmissionBase::evaluate()

template<typename V3, typename S3, typename DG>
  S3 SpecularTransmissionBase<V3,S3,DG>
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

