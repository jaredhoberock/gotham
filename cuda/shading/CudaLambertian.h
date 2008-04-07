/*! \file CudaLambertian.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a CUDA-compatible
 *         version of Lambertian.
 */

#pragma once

#include "../geometry/CudaDifferentialGeometry.h"
#include "../../shading/functions/LambertianBase.h"

typedef LambertianBase<float3,float3,CudaDifferentialGeometry> CudaLambertian;

