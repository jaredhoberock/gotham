/*! \file CudaScatteringDistributionFunction.inl
 *  \author Jared Hoberock
 *  \brief Inline file for CudaScatteringDistributionFunction.h.
 */

// XXX this is terribly shitty, but when compiled by anything
//     other than nvcc, we get compiler complaints about
//     dot product for CUDA vectors
#ifdef __CUDACC__

#include "CudaScatteringDistributionFunction.h"
#include "CudaHemisphericalEmission.h"
#include <vector_functions.h>
#include <stdcuda/vector_math.h>
#include "CudaLambertian.h"
#include "CudaPerspectiveSensor.h"
#include "CudaNullScattering.h"

//Spectrum CudaScatteringDistributionFunction
float3 CSDF
  ::evaluate(const float3 &wo,
             const CudaDifferentialGeometry &dg,
             const float3 &wi) const
{
  float3 result;

  // do a naive switch for now
  switch(mType)
  {
    case LAMBERTIAN:
    {
      const CudaLambertian &lambertian = (const CudaLambertian&)mFunction;
      result = lambertian.evaluate(wo,dg,wi);
      break;
    } // end LAMBERTIAN

    case NULL_SCATTERING:
    {
      const CudaNullScattering &null = (const CudaNullScattering&)mFunction;
      result = null.evaluate(wo,dg,wi);
      break;
    } // end NULL_SCATTERING

    default:
    {
      // XXX this should probably be a nan
      result = make_float3(1.0f,0.5f,0.25f);
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
      result = he.evaluate(wo,dg);
      break;
    } // end HEMISPHERICAL_EMISSION

    case PERSPECTIVE_SENSOR:
    {
      const CudaPerspectiveSensor &ps = (const CudaPerspectiveSensor&)mFunction;
      result = ps.evaluate(wo,dg);
      break;
    } // end PERSPECTIVE_SENSOR

    case NULL_SCATTERING:
    {
      const CudaNullScattering &ns = (const CudaNullScattering&)mFunction;
      result = ns.evaluate(wo,dg);
      break;
    } // end NULL_SCATTERING

    default:
    {
      // XXX this should probably be a nan
      result = make_float3(0.25f,0.5f,1.0f);
      break;
    } // end default
  } // end switch

  return result;
} // end CudaScatteringDistributionFunction::evaluate()

#endif // __CUDACC__

