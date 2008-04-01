/*! \file Fresnel.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of Fresnel class.
 */

#include "Fresnel.h"
#include "ScatteringDistributionFunction.h"

void *Fresnel
  ::operator new(size_t size, FunctionAllocator &alloc)
{
  return alloc.malloc();
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
  // we must figure out which ei / et is incident / transmittant
  bool entering = cosi > 0;
  float ei = mEtai, et = mEtat;
  if(!entering) std::swap(ei,et);

  float f = dielectric(ei, et, cosi);
  return Spectrum(f,f,f);
} // end FresnelDielectric::evaluate()

Spectrum FresnelDielectric
  ::evaluate(const float cosi, const float cost) const
{
  // we must figure out which ei / et is incident / transmittant
  // XXX we usually know the answer by this point
  //     we could pass the answer along to avoid this extra computation
  bool entering = cosi > 0;
  float ei = mEtai, et = mEtat;
  if(!entering) std::swap(ei,et);
  float f = dielectric(ei, et, cosi, cost);

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

