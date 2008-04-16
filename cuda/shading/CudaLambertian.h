/*! \file CudaLambertian.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a CUDA-compatible
 *         version of Lambertian.
 */

#pragma once

// XXX hack
#define inline inline __host__ __device__
#include "../geometry/CudaDifferentialGeometry.h"
#include "../../shading/functions/LambertianBase.h"
#undef inline

typedef LambertianBase<float3,float3,CudaDifferentialGeometry> CudaLambertian;

