/*! \file CudaScatteringDistributionFunction.inl
 *  \author Jared Hoberock
 *  \brief Inline file for CudaScatteringDistributionFunction.h.
 */

#include "CudaScatteringDistributionFunction.h"

Spectrum CudaScatteringDistributionFunction
  ::evaluate(const float3 &wo,
             const CudaDifferentialGeometry &dg,
             const float3 &wi) const
{
  Spectrum result;

  // do a naive switch for now
  switch(mType)
  {
    case LAMBERTIAN:
    {
      //result = Spectrum(1,1,1);
      result = dg.getNormal();
      break;
    } // end LAMBERTIAN

    default:
    {
      // XXX this should probably be a nan
      result = Spectrum(0,0,0);
      break;
    } // end default
  } // end switch

  return result;
} // end Spectrum::evaluate()

