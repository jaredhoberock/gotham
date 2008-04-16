/*! \file CudaHemisphericalEmission.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a device object
 *         equivalent to HemisphericalEmission.
 */

#pragma once

// XXX hack hack
#define inline __host__ __device__
#include "../geometry/CudaDifferentialGeometry.h"
#include "../../shading/functions/HemisphericalEmissionBase.h"
#undef inline

typedef HemisphericalEmissionBase<float3,float3,CudaDifferentialGeometry> CudaHemisphericalEmission;

