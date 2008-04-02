/*! \file CudaScatteringDistributionFunction.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a ScatteringDistributionFunction which
 *         can be used in device code.
 */

#pragma once

#include "../../shading/ScatteringFunctionBlock.h"
#include "../geometry/CudaDifferentialGeometry.h"

enum ScatteringType
{
  LAMBERTIAN,
  HEMISPHERICAL_EMISSION,
  NULL_SCATTERING
}; // end ScatteringType

//struct CudaScatteringDistributionFunction
// XXX BUG: give this a short name thanks to
//     this bug: 
//     http://forums.nvidia.com/index.php?showtopic=53767
struct CSDF
{
  ScatteringType mType;
  ScatteringFunctionBlock mFunction;

  inline __host__ __device__ float3 evaluate(const float3 &wo,
                                             const CudaDifferentialGeometry &dg,
                                             const float3 &wi) const;

  inline __host__ __device__ float3 evaluate(const float3 &wo,
                                             const CudaDifferentialGeometry &dg) const;
}; // end CudaScatteringDistributionFunction

typedef CSDF CudaScatteringDistributionFunction;

#include "CudaScatteringDistributionFunction.inl"

