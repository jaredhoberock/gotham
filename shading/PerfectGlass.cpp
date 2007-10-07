/*! \file PerfectGlass.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of PerfectGlass class.
 */

#include "PerfectGlass.h"

PerfectGlass
  ::PerfectGlass(const Spectrum &r,
                 const Spectrum &t,
                 const float etai,
                 const float etat)
    :Parent(),mReflectance(r),mTransmittance(t),mFresnel(etai,etat)
{
  ;
} // end PerfectGlass::PerfectGlass()

Spectrum PerfectGlass
  ::sampleReflectance(const Vector &wo,
                      const DifferentialGeometry &dg,
                      Vector &wi) const
{
  wi = dg.getNormal().reflect(wo);
  wi = wi.normalize();
  return evaluateReflectance(wo,dg);
} // end PerfectGlass::sample()

Spectrum PerfectGlass
  ::evaluateReflectance(const Vector &wo,
                        const DifferentialGeometry &dg) const
{
  float cosi = dg.getNormal().absDot(wo);
  return mReflectance * mFresnel.evaluate(cosi);
} // end PerfectGlass::evaluateReflectance()

Spectrum PerfectGlass
  ::sampleTransmittance(const Vector &wo,
                        const DifferentialGeometry &dg,
                        Vector &wi) const
{
  Spectrum result(0,0,0);

  // figure out which eta is incident/transmitted
  float cosi = wo.dot(dg.getNormal());
  bool entering = cosi > 0;
  float ei = mFresnel.mEtai, et = mFresnel.mEtat;
  if(!entering) std::swap(ei,et);

  // compute refracted ray direction
  float sini2 = 1.0f - cosi*cosi;
  float eta = ei / et;
  float sint2 = eta * eta * sini2;

  // check for total internal refraction
  if(sint2 <= 1.0f)
  {
    float cost = -sqrtf(std::max(0.0f, 1.0f - sint2));
    if(entering) cost = -cost;

    wi = (eta*cosi - cost)*dg.getNormal() - eta * wo;

    // compute fresnel term
    Spectrum f = mFresnel.evaluate(cosi, cost);
    result = (et*et)/(ei*ei) * (Spectrum::white() - f) * mTransmittance;
  } // end if

  return result;
} // end PerfectGlass::sampleTransmittance()

Spectrum PerfectGlass
  ::evaluateTransmittance(const Vector &wo,
                          const DifferentialGeometry &dg) const
{
  Spectrum result(0,0,0);

  // figure out which eta is incident/transmitted
  float cosi = wo.dot(dg.getNormal());
  bool entering = cosi > 0;
  float ei = mFresnel.mEtai, et = mFresnel.mEtat;
  if(!entering) std::swap(ei,et);

  // compute refracted ray direction
  float sini2 = 1.0f - cosi*cosi;
  float eta = ei / et;
  float sint2 = eta * eta * sini2;

  // check for total internal refraction
  if(sint2 <= 1.0f)
  {
    float cost = -sqrtf(std::max(0.0f, 1.0f - sint2));
    if(entering) cost = -cost;

    // compute fresnel term
    Spectrum f = mFresnel.evaluate(cosi, cost);
    result = (et*et)/(ei*ei) * (Spectrum::white() - f) * mTransmittance;
  } // end if

  return result;
} // end PerfectGlass::evaluateTransmittance()

Spectrum PerfectGlass
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
  Spectrum result;
  delta = true;
  
  Vector wr;
  Spectrum reflection = sampleReflectance(wo, dg, wr);
  Vector wt;
  Spectrum transmission = sampleTransmittance(wo, dg, wt);

  float pReflection = reflection.luminance();
  pReflection /= (pReflection + transmission.luminance());
  if(u2 < pReflection)
  {
    result = reflection;
    wi = wr;
    pdf = pReflection;
    component = 0;
  } // end if
  else
  {
    result = transmission;
    wi = wt;
    pdf = 1.0f - pReflection;
    component = 1;
  } // end else

  return result / dg.getNormal().absDot(wi);
} // end PerfectGlass::sample()

float PerfectGlass
  ::evaluatePdf(const Vector &wo,
                const DifferentialGeometry &dg,
                const Vector &wi,
                const bool delta,
                const ComponentIndex component) const
{
  if(component > 1) return 0;

  // evaluate both functions
  Spectrum reflection = evaluateReflectance(wo, dg);
  Spectrum transmission = evaluateReflectance(wo, dg);

  float pReflection = reflection.luminance();
  pReflection /= (pReflection + transmission.luminance());

  // which pdf are we interested in?
  return (component == 0) ? pReflection : 1.0f - pReflection;
} // end PerfectGlass::evaluatePdf()

