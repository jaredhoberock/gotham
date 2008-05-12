/*! \file kajiyaPathTracerUtil.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to some CUDA kernels
 *         used by the CudaKajiyaPathTracer class.
 */

#pragma once

#include <stdcuda/device_types.h>
#include "../geometry/CudaDifferentialGeometry.h"
#include "../geometry/CudaDifferentialGeometryArray.h"
#include "../primitives/CudaIntersection.h"

extern "C" void updateThroughput(const stdcuda::device_ptr<const float3> &scattering,
                                 const stdcuda::device_ptr<const float> &pdfs,
                                 const stdcuda::device_ptr<const bool> &stencil,
                                 const stdcuda::device_ptr<float3> &throughput,
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

extern "C" void stratify(const unsigned int w, const unsigned int h,
                         const stdcuda::device_ptr<float4> &u,
                         const size_t n);

extern void flipVectors(const stdcuda::device_ptr<const float3> &src,
                        const stdcuda::device_ptr<const bool> &stencil,
                        const stdcuda::device_ptr<float3> &dst,
                        const size_t n);

void createShadowRays(const stdcuda::device_ptr<const float3> &from,
                      const stdcuda::device_ptr<const float3> &to,
                      const float epsilon,
                      const stdcuda::device_ptr<const bool> &stencil,
                      const stdcuda::device_ptr<float3> &directions,
                      const size_t n);

void evaluateGeometricTerm(const stdcuda::device_ptr<const float3> &x0,
                           const stdcuda::device_ptr<const float3> &n0,
                           const stdcuda::device_ptr<const float3> &x1,
                           const stdcuda::device_ptr<const float3> &n1,
                           const stdcuda::device_ptr<const bool> &stencil,
                           const stdcuda::device_ptr<float> &g,
                           const size_t n);

void toProjectedSolidAnglePdf(const stdcuda::device_ptr<const float3> &wo,
                              const stdcuda::device_ptr<const float3> &normals,
                              const stdcuda::device_ptr<const bool> &stencil,
                              const stdcuda::device_ptr<float> &pdf,
                              const size_t n);

