/*! \file CudaDifferentialGeometry.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a DifferentialGeometry
 *         class which can be used in CUDA code.
 */

#pragma once

#include <host_defines.h>
#include <vector_types.h>

// XXX hack hack
#define inline inline __host__ __device__
#include "../../geometry/DifferentialGeometryBase.h"
#undef inline

typedef DifferentialGeometryBase<float3, float3, float2, float3> CudaDifferentialGeometry;

