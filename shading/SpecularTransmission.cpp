/*! \file SpecularTransmission.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of SpecularTransmission class.
 */

#include "SpecularTransmission.h"
#include "../geometry/Mappings.h"

SpecularTransmission
  ::SpecularTransmission(const Spectrum &t,
                         const float etai,
                         const float etat)
    :Parent0(),Parent1(t,etai,etat)
{
  ;
} // end SpecularTransmission::SpecularTransmission()

Spectrum SpecularTransmission
  ::sample(const Vector &wo,
           const DifferentialGeometry &dg,
           const float u0,
           const float u1,
           const float u2,
           Vector &wi,
           float &pdf,
           bool &delta,
           ComponentIndex &component) const
{
  return Parent1::sample(wo,dg.getNormal(),wi,pdf,delta,component);
} // end SpecularTransmission::sample()

Spectrum SpecularTransmission
  ::evaluate(const Vector &wo,
             const DifferentialGeometry &dg,
             const Vector &wi) const
{
  return Parent1::evaluate();
} // end SpecularTransmission::evaluate()

Spectrum SpecularTransmission
  ::evaluate(const Vector &wo,
             const DifferentialGeometry &dg,
             const Vector &wi,
             const bool delta,
             const ComponentIndex component,
             float &pdf) const
{
  pdf = 0;
  Spectrum result(Spectrum::black());
  if(delta)
  {
    pdf = 1.0;

    // figure out which eta is incident/transmitted
    float cosi = wo.dot(dg.getNormal());
    bool entering = cosi > 0;
    float ei = mEtai, et = mEtat;
    if(!entering) std::swap(ei,et);

    // compute refracted ray direction
    float sini2 = 1.0f - cosi*cosi;
    float eta = ei / et;
    float sint2 = eta * eta * sini2;

    // assume no total internal refraction
    float cost = -sqrtf(std::max(0.0f, 1.0f - sint2));
    if(entering) cost = -cost;

    // compute fresnel term
    float fresnel = evaluateFresnelDielectric(ei, et, cosi, cost);
    Spectrum f = Spectrum::white() - Spectrum(fresnel,fresnel,fresnel);
    f.saturate();
    result = (et*et)/(ei*ei) * f * mTransmittance;

    result /= dg.getNormal().absDot(wi);
  } // end if

  return result;
} // end SpecularTransmission::evaluate()

