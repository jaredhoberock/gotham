/*! \file CudaDifferentialGeometry.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a DifferentialGeometry
 *         class which can be used in CUDA code.
 */

#pragma once

#include "../../geometry/DifferentialGeometry.h"

typedef DifferentialGeometryBase<float3, float3, float2, float3> CudaDifferentialGeometry;

