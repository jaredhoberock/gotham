/*! \file CudaDifferentialGeometryArray.h
 *  \author Jared Hoberock
 *  \brief Defines a CUDA interface to DifferentialGeometryArray.
 */

#pragma once

#include <stdcuda/stdcuda.h>

// XXX hack total
#define inline inline __host__ __device__
#include "DifferentialGeometryArrayBase.h"
#undef inline

typedef DifferentialGeometryArrayBase<float3,float3,float2,float3> CudaDifferentialGeometryArray;

