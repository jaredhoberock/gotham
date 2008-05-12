/*! \file cudaDebugRendererUtil.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to some helper functions
 *         for the CudaDebugRenderer class.
 */

#pragma once

#include <stdcuda/device_types.h>
#include "../geometry/CudaDifferentialGeometry.h"

void sumScatteringAndEmission(const stdcuda::device_ptr<const float3> &scattering,
                              const stdcuda::device_ptr<const float3> &emission,
                              const stdcuda::device_ptr<const bool> &stencil,
                              const stdcuda::device_ptr<float3> &result,
                              const size_t n);

void flipVectors(const stdcuda::device_ptr<const float3> &src,
                 const stdcuda::device_ptr<float3> &result,
                 const size_t n);

