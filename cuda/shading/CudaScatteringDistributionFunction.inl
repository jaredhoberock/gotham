/*! \file CudaScatteringDistributionFunction.inl
 *  \author Jared Hoberock
 *  \brief Inline file for CudaScatteringDistributionFunction.h.
 */

#include "CudaScatteringDistributionFunction.h"
#include "CudaLambertian.h"
#include "CudaHemisphericalEmission.h"
#include <vector_functions.h>

//Spectrum CudaScatteringDistributionFunction
float3 CSDF
  ::evaluate(const float3 &wo,
             const CudaDifferentialGeometry &dg,
             const float3 &wi) const { float3 result;

  // do a naive switch for now
  switch(mType)
  {
    case LAMBERTIAN:
    {
      const CudaLambertian &lambertian = (const CudaLambertian&)mFunction;
      const float3 &albedo = (const float3&)lambertian.mAlbedoOverPi;
      result = albedo;
      break;
    } // end LAMBERTIAN

    default:
    {
      // XXX this should probably be a nan
      result = make_float3(0,0,0);
      break;
    } // end default
  } // end switch

  return result;
} // end Spectrum::evaluate()

//Spectrum CudaScatteringDistributionFunction
float3 CSDF
  ::evaluate(const float3 &wo,
             const CudaDifferentialGeometry &dg) const
{
  float3 result;

  // do a naive switch for now
  switch(mType)
  {
    case HEMISPHERICAL_EMISSION:
    {
      const CudaHemisphericalEmission &he = (const CudaHemisphericalEmission&)mFunction;
      const float3 &r = (const float3&)he.mRadiance;
      result = r;
      break;
    } // end HEMISPHERICAL_EMISSION

    default:
    {
      // XXX this should probably be a nan
      result = make_float3(0,0,0);
      break;
    } // end default
  } // end switch

  return result;
} // end CudaScatteringDistributionFunction::evaluate()

