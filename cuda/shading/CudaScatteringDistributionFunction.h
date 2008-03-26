/*! \file CudaScatteringDistributionFunction.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a ScatteringDistributionFunction which
 *         can be used in device code.
 */

#pragma once

#include "../../shading/ScatteringFunctionBlock.h"
#include "../geometry/CudaDifferentialGeometry.h"

// this defines the CUDA vector types
#include <vector_types.h>

enum ScatteringType
{
  LAMBERTIAN
}; // end ScatteringType

struct CudaScatteringDistributionFunction
{
  ScatteringFunctionBlock mFunction;
  ScatteringType mType;

  inline __host__ __device__ Spectrum evaluate(const float3 &wo,
                                               const CudaDifferentialGeometry &dg,
                                               const float3 &wi) const;
}; // end CudaScatteringDistributionFunction

#include "CudaScatteringDistributionFunction.inl"

