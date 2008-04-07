/*! \file evaluateUnidirectionalScattering.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a CUDA kernel which
 *         evaluates the unidirectional scattering of a
 *         list of bsdf jobs.
 */

#pragma once

#include "../geometry/CudaDifferentialGeometry.h"
#include "CudaScatteringDistributionFunction.h"
#include <spectrum/Spectrum.h>

extern "C" void evaluateUnidirectionalScattering(const CudaScatteringDistributionFunction *f,
                                                 const float3 *wo,
                                                 const CudaDifferentialGeometry *dg,
                                                 const int *stencil,
                                                 float3 *results,
                                                 const size_t n);


extern "C" void evaluateUnidirectionalScatteringStride(const CudaScatteringDistributionFunction *f,
                                                       const float3 *wo,
                                                       const CudaDifferentialGeometry *dg,
                                                       const size_t dgStride,
                                                       const int *stencil,
                                                       float3 *results,
                                                       const size_t n);


