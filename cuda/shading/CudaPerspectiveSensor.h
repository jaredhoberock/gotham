/*! \file CudaPerspectiveSensor.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a CUDA compatible
 *         version of PerspectiveSensor.
 */

#pragma once

// XXX hack
#define inline inline __host__ __device__
#include "../geometry/CudaDifferentialGeometry.h"
#include "../../shading/functions/PerspectiveSensorBase.h"
#undef inline

typedef PerspectiveSensorBase<float3,float3,CudaDifferentialGeometry> CudaPerspectiveSensor;

