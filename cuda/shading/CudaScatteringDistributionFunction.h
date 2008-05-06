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
  PERSPECTIVE_SENSOR,
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

  inline __host__ __device__ void sample(const CudaDifferentialGeometry &dg,
                                         const float u0,
                                         const float u1,
                                         const float u2,
                                         float3 &s,
                                         float3 &wo,
                                         float &pdf,
                                         bool &delta) const;

  inline __host__ __device__ void sample(const float3 &wo,
                                         const CudaDifferentialGeometry &dg,
                                         const float u0,
                                         const float u1,
                                         const float u2,
                                         float3 &s,
                                         float3 &wi,
                                         float &pdf,
                                         bool &delta,
                                         unsigned int &component) const;
                                         
}; // end CudaScatteringDistributionFunction

typedef CSDF CudaScatteringDistributionFunction;

#include "CudaScatteringDistributionFunction.inl"

