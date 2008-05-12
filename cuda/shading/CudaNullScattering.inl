/*! \file CudaNullScattering.inl
 *  \author Jared Hoberock
 *  \brief Inline file for CudaNullScattering.h.
 */

#include "CudaNullScattering.h"
#include <vector_functions.h>

float3 CudaNullScattering
  ::evaluate(const float3 &wo,
             const CudaDifferentialGeometry &dg,
             const float3 &wi) const
{
  return make_float3(0,0,0);
} // end CudaDifferentialGeometry::evaluate()

float3 CudaNullScattering
  ::evaluate(void) const
{
  return make_float3(0,0,0);
} // end CudaDifferentialGeometry::evaluate()

float3 CudaNullScattering
  ::evaluate(const float3 &wo,
             const CudaDifferentialGeometry &dg) const
{
  return make_float3(0,0,0);
} // end CudaDifferentialGeometry::evaluate()

