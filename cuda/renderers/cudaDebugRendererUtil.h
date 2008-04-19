/*! \file cudaDebugRendererUtil.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to some helper functions
 *         for the CudaDebugRenderer class.
 */

#pragma once

#include <stdcuda/device_types.h>
#include "../geometry/CudaDifferentialGeometry.h"

extern "C"
  void sampleEyeRaysEpilogue(const stdcuda::device_ptr<const CudaDifferentialGeometry> &dg,
                             const stdcuda::device_ptr<float4> &originsAndMinT,
                             const stdcuda::device_ptr<float4> &directionsAndMaxT,
                             const size_t n);

extern "C"
  void rayDirectionsToWo(const stdcuda::device_ptr<const float4> &directionsAndMaxT,
                         const stdcuda::device_ptr<float3> &wo,
                         const size_t n);

extern "C"
  void sumScatteringAndEmission(const stdcuda::device_ptr<const float3> &scattering,
                                const stdcuda::device_ptr<const float3> &emission,
                                const stdcuda::device_ptr<const int> &stencil,
                                const stdcuda::device_ptr<float3> &result,
                                const size_t n);

