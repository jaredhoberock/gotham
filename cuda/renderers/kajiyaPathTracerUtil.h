/*! \file kajiyaPathTracerUtil.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to some CUDA kernels
 *         used by the CudaKajiyaPathTracer class.
 */

#pragma once

#include <stdcuda/device_types.h>
#include "../geometry/CudaDifferentialGeometry.h"
#include "../primitives/CudaIntersection.h"

extern "C" void differentialGeometryToRayOrigins(const stdcuda::device_ptr<const CudaDifferentialGeometry> &dg,
                                                 const stdcuda::device_ptr<const bool> &stencil,
                                                 const stdcuda::device_ptr<float4> &originsAndMinT,
                                                 const size_t n);

extern "C" void finalizeRays(const stdcuda::device_ptr<const CudaDifferentialGeometry> &dg,
                             const float minT,
                             const float maxT,
                             const stdcuda::device_ptr<const bool> &stencil,
                             const stdcuda::device_ptr<float4> &originsAndMinT,
                             const stdcuda::device_ptr<float4> &directionsAndMaxT,
                             const size_t n);


extern "C" void updateThroughput(const stdcuda::device_ptr<const float3> &scattering,
                                 const stdcuda::device_ptr<const float> &pdfs,
                                 const stdcuda::device_ptr<const bool> &stencil,
                                 const stdcuda::device_ptr<float3> &throughput,
                                 const size_t n);

extern "C" void createShadowRays(const stdcuda::device_ptr<const CudaIntersection> &intersections,
                                 const stdcuda::device_ptr<const CudaDifferentialGeometry> &dg,
                                 const float epsilon,
                                 const stdcuda::device_ptr<const bool> &stencil,
                                 const stdcuda::device_ptr<float4> &originsAndMinT,
                                 const stdcuda::device_ptr<float4> &directionsAndMaxT,
                                 const size_t n);

extern "C" void flipRayDirections(const stdcuda::device_ptr<const float4> &directionsAndMaxT,
                                  const stdcuda::device_ptr<const bool> &stencil,
                                  const stdcuda::device_ptr<float3> &wo,
                                  const size_t n);

extern "C" void toProjectedSolidAnglePdf(const stdcuda::device_ptr<const float3> &wo,
                                         const stdcuda::device_ptr<const CudaDifferentialGeometry> &dg,
                                         const stdcuda::device_ptr<const bool> &stencil,
                                         const stdcuda::device_ptr<float> &pdf,
                                         const size_t n);

extern "C" void rayDirectionsToFloat3(const stdcuda::device_ptr<const float4> &directionsAndMaxT,
                                      const stdcuda::device_ptr<const bool> &stencil,
                                      const stdcuda::device_ptr<float3> &w,
                                      const size_t n);

extern "C" void evaluateGeometricTerm(const stdcuda::device_ptr<const CudaDifferentialGeometry> &dg0,
                                      const size_t dg0Stride,
                                      const stdcuda::device_ptr<const CudaDifferentialGeometry> &dg1,
                                      const stdcuda::device_ptr<const bool> &stencil,
                                      const stdcuda::device_ptr<float> &g,
                                      const size_t n);

extern "C" void accumulateLightSample(const float weight,
                                      const stdcuda::device_ptr<const float3> &throughput,
                                      const stdcuda::device_ptr<const float3> &scattering,
                                      const stdcuda::device_ptr<const float3> &emission,
                                      const stdcuda::device_ptr<const float> &geometricTerm,
                                      const stdcuda::device_ptr<const float> &pdf,
                                      const stdcuda::device_ptr<const bool> &stencil,
                                      const stdcuda::device_ptr<float3> &result,
                                      const size_t n);

extern "C" void extractFloat2(const stdcuda::device_ptr<const float4> &x,
                              const stdcuda::device_ptr<float2> &y,
                              const size_t n);

extern "C" void accumulateEmission(const stdcuda::device_ptr<const float3> &throughput,
                                   const stdcuda::device_ptr<const float3> &emission,
                                   const stdcuda::device_ptr<const bool> &stencil,
                                   const stdcuda::device_ptr<float3> &result,
                                   const size_t n);

extern "C" void divideByPdf(const stdcuda::device_ptr<const float> &pdf,
                            const stdcuda::device_ptr<float3> &throughput,
                            const size_t n);

extern "C" void flipVectors(const stdcuda::device_ptr<float3> &w,
                            const stdcuda::device_ptr<const bool> &stencil,
                            const size_t n);

