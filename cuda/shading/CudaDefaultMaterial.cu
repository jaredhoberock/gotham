/*! \file CudaDefaultMaterial.cu
 *  \author Jared Hoberock
 *  \brief Implementation of CudaDefaultMaterial class.
 */

#define inline inline __host__ __device__

#include "CudaDefaultMaterial.h"
#include "CudaScatteringDistributionFunction.h"
#include "../include/CudaShadingInterface.h"
#include <stdcuda/fill_if.h>
using namespace stdcuda;

#undef inline

const char *CudaDefaultMaterial
  ::getName(void) const
{
  return "CudaDefaultMaterial";
} // end CudaDefaultMaterial::getName()

void CudaDefaultMaterial
  ::evaluateScattering(CudaShadingInterface &context,
                       const device_ptr<const CudaDifferentialGeometry> &dg,
                       const size_t dgStride,
                       const device_ptr<const int> &stencil,
                       const device_ptr<CudaScatteringDistributionFunction> &f,
                       const size_t n) const
{
  CudaScatteringDistributionFunction value;

  // create something white
  context.diffuse(make_float3(1,1,1), value);

  fill_if(f.get(), f.get() + n, stencil.get(), value);
} // end CudaMaterial::evaluateScattering()

