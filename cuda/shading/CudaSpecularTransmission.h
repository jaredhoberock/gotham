/*! \file CudaSpecularTransmission.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a CUDA-compatible
 *         version of SpecularTransmission.
 */

#pragma once

// XXX hack
#define inline inline __host__ __device__
#include "../geometry/CudaDifferentialGeometry.h"
#include "../../shading/functions/SpecularTransmissionBase.h"
#undef inline

typedef SpecularTransmissionBase<float3,float3,CudaDifferentialGeometry> CudaSpecularTransmission;

