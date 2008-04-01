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
                                                 Spectrum *results,
                                                 const size_t n);


