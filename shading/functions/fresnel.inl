/*! \file fresnel.inl
 *  \author Jared Hoberock
 *  \brief Inline file for fresnel.h.
 */

#include "fresnel.h"

float evaluateFresnelDielectric(const float ei,
                                const float et,
                                const float cosi)
{
  // compute cost and pass along
  
  // start with sint
  float sint = 1.0f - cosi*cosi;
  sint = sint < 0 ? 0 : sint;
  sint = (ei/et) * sqrtf(sint);

  // handle total internal reflection
  if(sint > 1) return 1;

  float cost = 1.0f - sint*sint;
  cost = cost < 0 ? 0 : cost;
  cost = sqrtf(cost);

  return evaluateFresnelDielectric(ei,et,fabs(cosi),cost);
} // end evaluateFresnelDielectric()

float evaluateFresnelDielectric(const float ei,
                                const float et,
                                const float cosi,
                                const float cost)
{
  float rParallel = ((et * cosi) - (ei * cost)) /
                    ((et * cosi) + (ei * cost));

  float rPerpendicular = ((ei * cosi) - (et * cost)) /
                         ((ei * cosi) + (et * cost));

  float result = (rParallel * rParallel + rPerpendicular * rPerpendicular) / 2.0f;

  // clamp to 1
  result = result > 1.0f ? 1.0f : result;

  return result;
} // end evaluateFresnelDielectric()

float evaluateFresnelDielectricUnknownOrientation(float ei,
                                                  float et,
                                                  const float cosi)
{
  // figure out if the ray is entering or exiting the dielectric
  if(cosi <= 0)
  {
    // swap indices
    float temp = ei;
    ei = et;
    et = temp;
  } // end if

  return evaluateFresnelDielectric(ei,et,cosi);
} // end evaluateFresnelDielectric()

template<typename S3>
  S3 approximateFresnelAbsorptionCoefficient(const S3 &r)
{
  S3 result = r;

  result.x /= (1.0f - result.x);
  result.y /= (1.0f - result.y);
  result.z /= (1.0f - result.z);

  // saturate before the square root
  result = saturate(result);

  result.x = sqrtf(result.x);
  result.y = sqrtf(result.y);
  result.z = sqrtf(result.z);

  result.x *= 2.0f;
  result.y *= 2.0f;
  result.z *= 2.0f;

  return result;
} // end approximateFresnelAbsorptionCoefficient()

template<typename S3>
  inline S3 evaluateFresnelConductor(const S3 &absorption,
                                     const float eta,
                                     const float cosi)
{
  float c2 = cosi*cosi;

  S3 eta2;
  eta2.x = eta;
  eta2.y = eta;
  eta2.z = eta;

  S3 twoEtaCosi = 2.0f * cosi * eta2;

  // find the square of eta
  eta2 *= eta2;

  S3 absorption2 = absorption*absorption;

  S3 cosi2;
  cosi2.x = c2;
  cosi2.y = c2;
  cosi2.z = c2;
  S3 tmp = cosi2 * (eta2 + absorption2);

  S3 one;
  one.x = 1;
  one.y = 1;
  one.z = 1;

  S3 rParallel2 = (tmp - twoEtaCosi + one) /
                  (tmp + twoEtaCosi + one);
  S3 tmpf = eta2 + absorption2;

  S3 rPerp2 = (tmpf - twoEtaCosi + cosi2) /
              (tmpf + twoEtaCosi + cosi2);

  return (rParallel2 + rPerp2) / 2.0f;
} // end evaluateFresnelConductor()

