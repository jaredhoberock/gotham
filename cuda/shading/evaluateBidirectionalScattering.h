/*! \file evaluateBidirectionalScattering.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a CUDA kernel which
 *         evaluates the bidirectional scattering of a
 *         list of bsdf jobs.
 */

#pragma once

#include "../geometry/CudaDifferentialGeometry.h"
#include "CudaScatteringDistributionFunction.h"
#include "../../include/Spectrum.h"

extern "C" void evaluateBidirectionalScattering(const CudaScatteringDistributionFunction *f,
                                                const float3 *wo,
                                                const CudaDifferentialGeometry *dg,
                                                const float3 *wi,
                                                const bool *stencil,
                                                float3 *results,
                                                const size_t n);

extern "C" void evaluateBidirectionalScatteringStride(const CudaScatteringDistributionFunction *f,
                                                      const float3 *wo,
                                                      const CudaDifferentialGeometry *dg,
                                                      const size_t dgStride,
                                                      const float3 *wi,
                                                      const bool *stencil,
                                                      float3 *results,
                                                      const size_t n);

