/*! \file PhongTransmission.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of PhongTransmission class.
 *         XXX TODO: generalize the distribution
 */

#include "PhongTransmission.h"
#include "../geometry/Mappings.h"

PhongTransmission
  ::PhongTransmission(const Spectrum &t,
                      const float etai,
                      const float etat,
                      const float exponent)
    :mFresnel(etai,etat),mTransmittance(t),
     mTransmittanceOverTwoPi(mTransmittance * INV_TWOPI),
     mExponent(exponent)
{
  ;
} // end PhongTransmission::PhongTransmission()

Spectrum PhongTransmission
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

  Spectrum result(0,0,0);
  pdf = 0;

  // sample a microfacet normal
  Vector m;
  Mappings::unitSquareToPhongLobe(u0, u1, dg.getNormal(), mExponent,
                                  dg.getTangent(), dg.getBinormal(), dg.getNormal(),
                                  m, pdf);

  // we are able to sample the Phong distribution exactly
  float D = pdf;

  // compute the transmitted direction
  float cosi = wo.dot(m);
  bool entering = wo.dot(dg.getNormal()) > 0;
  float ei = mFresnel.mEtai, et = mFresnel.mEtat;
  if(!entering) std::swap(ei,et);

  // compute refracted ray direction
  float sini2 = 1.0f - cosi*cosi;
  float eta = ei / et;
  float sint2 = eta * eta * sini2;

  // check for total internal refraction
  if(sint2 <= 1.0f)
  {
    // from Walter et al, 2007, equation 24
    // stated without proof, so I don't really believe it
    //pdf *= dg.getNormal().absDot(m);

    float cost = -sqrtf(std::max(0.0f, 1.0f - sint2));
    if(entering) cost = -cost;

    // Walter et al, 2007, equation 40
    wi = (eta * cosi - cost) * m - eta * wo;

    // compute fresnel term
    Spectrum F = mFresnel.evaluate(cosi, cost);

    // compute geometry term
    float G = 1.0f;

    // Walter et al, 2007, equation 17
    // i believe the et^2 term from the paper is incorrect
    // rather, it is part of only the bsdf, not also the pdf
    float J = m.absDot(wi);
    float d = (ei*wo.dot(m) + et*wi.dot(m));
    J /= d*d;

    // Walter et al, 2007, equation 38
    // XXX this seems unnecessary, because we end up multiplying it
    //     into the result, which we will divide by pdf later
    pdf *= J;

    // Walter et al, 2007, equation 21
    result = m.absDot(wo) * (Spectrum::white() - F) * G * D * J;
    result /= (dg.getNormal().absDot(wo) * dg.getNormal().absDot(wi));

    // for reciprocity
    result *= (et*et);
    result /= (ei*ei);
  } // end if

  return result;
} // end PhongTransmission::sample()

float PhongTransmission
  ::evaluatePdf(const Vector3 &wo,
                const DifferentialGeometry &dg,
                const Vector3 &wi) const
{
  bool entering = wo.dot(dg.getNormal()) > 0;
  float ei = mFresnel.mEtai, et = mFresnel.mEtat;
  if(!entering) std::swap(ei,et);

  // wo & wi must lie in different hemispheres
  bool exiting = wi.dot(dg.getNormal()) < 0;
  if(entering != exiting) return 0;

  // compute the halfangle
  Vector m = -(ei * wo + et * wi);
  m = m.normalize();

  float J = m.absDot(wi);
  float d = (ei*wo.dot(m) + et*wi.dot(m));
  J /= d*d;

  // evaluate the phong distribution and multiply by the Jacobian
  // XXX hide the phong distribution evaluation somewhere else
  return (mExponent + 2.0f) * powf(dg.getNormal().absDot(m), mExponent) * INV_TWOPI * J;
} // end PhongTransmission::evaluatePdf()

Spectrum PhongTransmission
  ::evaluate(const Vector &wo,
             const DifferentialGeometry &dg,
             const Vector &wi) const
{
  Spectrum result(Spectrum::black());

  bool entering = wo.dot(dg.getNormal()) > 0;
  float ei = mFresnel.mEtai, et = mFresnel.mEtat;
  if(!entering) std::swap(ei,et);

  // wo & wi must lie in different hemispheres
  bool exiting = wi.dot(dg.getNormal()) < 0;
  if(entering != exiting) return Spectrum::black();

  // compute the halfangle
  Vector m = -(ei * wo + et * wi);
  m = m.normalize();

  // evaluate the phong distribution
  // XXX hide the phong distribution evaluation somewhere else
  float D = (mExponent + 2.0f) * powf(dg.getNormal().absDot(m), mExponent) * INV_TWOPI;

  // compute the transmitted direction
  float cosi = wo.dot(m);

  // compute refracted ray direction
  float sini2 = 1.0f - cosi*cosi;
  float eta = ei / et;
  float sint2 = eta * eta * sini2;

  float cost = -sqrtf(std::max(0.0f, 1.0f - sint2));
  if(entering) cost = -cost;

  // compute fresnel term
  Spectrum F = mFresnel.evaluate(cosi, cost);

  // compute geometry term
  float G = 1.0f;

  // Walter et al, 2007, equation 17
  // i believe the et^2 term from the paper is incorrect
  // rather, it is part of only the bsdf, not also the pdf
  float J = m.absDot(wi);
  float d = (ei*wo.dot(m) + et*wi.dot(m));
  J /= d*d;

  // Walter et al, 2007, equation 21
  result = m.absDot(wo) * (Spectrum::white() - F) * G * D * J;
  result /= (dg.getNormal().absDot(wo) * dg.getNormal().absDot(wi));

  // for reciprocity
  result *= (et*et);
  result /= (ei*ei);

  return result;
} // end PhongTransmission::evaluate()


