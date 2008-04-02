/*! \file CudaHemisphericalEmission.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a device object
 *         equivalent to HemisphericalEmission.
 */

#pragma once

#include "../../shading/functions/HemisphericalEmissionBase.h"
typedef HemisphericalEmissionBase<float3,float3,CudaDifferentialGeometry> CudaHemisphericalEmission;

