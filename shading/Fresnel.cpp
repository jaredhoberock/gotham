/*! \file Fresnel.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of Fresnel class.
 */

#include "Fresnel.h"
#include "ScatteringDistributionFunction.h"

void *Fresnel
  ::operator new(unsigned int size)
{
  return ScatteringDistributionFunction::mPool.malloc();
} // end Fresnel::operator new()

FresnelDielectric
  ::FresnelDielectric(const float ei, const float et)
    :mEtai(ei),mEtat(et)
{
  ;
}; // end FresnelDielectric::FresnelDielectric()

Spectrum FresnelDielectric
  ::evaluate(const float cosi) const
{
  float f = dielectric(mEtai, mEtat, cosi);
  return Spectrum(f,f,f);
} // end FresnelDielectric::evaluate()

Spectrum FresnelDielectric
  ::evaluate(const float cosi, const float cost) const
{
  float f = dielectric(mEtai, mEtat, cosi, cost);
  return Spectrum(f,f,f);
} // end FresnelDielectric::evaluate()

FresnelConductor
  ::FresnelConductor(const Spectrum &e, const Spectrum &a)
    :mEta(e),mAbsorbance(a)
{
  ;
} // end FresnelConductor::FresnelConductor()

Spectrum FresnelConductor
  ::evaluate(const float cosi) const
{
  return conductor(fabs(cosi), mEta, mAbsorbance);
} // end FresnelConductor::evaluate()

